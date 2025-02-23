import json
import time
from abc import ABC
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import ray
import torch
import torch.nn as nn
from tqdm import tqdm

from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray

logger = init_logger(__name__)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None

    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    prompts: the prompts used to generate responses
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    prompts: list[str]
    prompt_metadata: list[dict]


class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,
        critic: nn.Module,
        reward_model: nn.Module,
        initial_model: Actor,
        tokenizer,
        prompt_max_len: int,
        kl_controller,
        strategy=None,
        remote_rm_url: list[str] = None,
        reward_fn=None,
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.initial_model = initial_model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator

        # custom reward func for reinforced finetuning
        self.custom_reward_func = None
        if remote_rm_url and remote_rm_url[0].endswith(".py"):
            print(f"Loading custom `reward_func(queries, prompts)` from {remote_rm_url[0]}")
            import importlib.util
            import inspect

            spec = importlib.util.spec_from_file_location("reward_func", remote_rm_url[0])
            reward_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(reward_module)
            self.custom_reward_func = reward_module.reward_func

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, extra_rm_args, all_prompts: Union[Dict, List[Dict]], **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        args = self.strategy.args
        # generate responses
        samples_list = self.generate_samples(all_prompts, **generate_kwargs)
        torch.distributed.barrier()

        experiences = []
        for samples in tqdm(
            samples_list,
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            experiences.append(self.make_experience(extra_rm_args, samples).to_device("cpu"))

        experiences, rewards = self.process_experiences(experiences)

        # calculate return and advantages
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.
        """
        all_prompts, all_prompt_metadata = all_prompts
        assert not getattr(self, "packing_samples", False)
        args = self.strategy.args
        self.actor.eval()
        # sample multiple response
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_prompt_metadata = sum([[deepcopy(prompt_metadata) for _ in range(args.n_samples_per_prompt)] for prompt_metadata in all_prompt_metadata], [])
        samples_list = []
        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
            prompt_metadata = all_prompt_metadata[i : i + args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
            samples = Samples(
                sequences=sequences,
                attention_mask=attention_mask,
                action_mask=action_mask,
                num_actions=action_mask.size(1),
                packed_seq_lens=None,
                response_length=action_mask.float().sum(dim=-1),
                total_length=attention_mask.float().sum(dim=-1),
                prompts=prompt_strings,
                prompt_metadata=prompt_metadata,
            )
            samples_list.append(samples)
        return samples_list

    @torch.no_grad()
    def make_experience(self, extra_rm_args, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        if self.initial_model is not None:
            self.initial_model.eval()
        if self.reward_model is not None:
            self.reward_model.eval()
        if self.critic is not None:
            self.critic.eval()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask)

        # init log probs
        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask) if self.initial_model is not None else None

        # values
        if self.critic is not None:
            value = self.critic(sequences, num_actions, attention_mask)
        else:
            value = None

        # rewards
        if self.remote_rm_url is not None:
            # remote RM
            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
            if self.custom_reward_func:
                r = self.custom_reward_func(queries, samples.prompts, samples.prompt_metadata, **extra_rm_args).to(device=action_log_probs.device)
            else:
                r = remote_rm_fn(self.remote_rm_url, queries=queries, prompts=samples.prompts).to(
                    device=action_log_probs.device
                )
        else:
            # local RM
            r = self.reward_model(sequences, attention_mask)

        if self.initial_model is not None:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs)

        info = {
            "kl": masked_mean(kl, action_mask, dim=-1),
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }

        if type(r) == dict:
            assert 'reward' in r
            info.update(r)
        else:
            info['reward'] = r

        # reset model state
        self.actor.train()
        if self.critic is not None:
            self.critic.train()

        return Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience]) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        args = self.strategy.args
        # reward shaping for RLOO
        if args.advantage_estimator == "rloo":
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            rewards = rewards - baseline
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences))
            return experiences, rewards
        # default rewards
        return experiences, [experience.info["reward"] for experience in experiences]

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function that computes advantages and returns from rewards and values.
        Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
        Note that rewards may include a KL divergence loss term.

        Advantages looks like this:
        Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
              - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Returns looks like this:
        Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                   + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

        Input:
        - values: Tensor of shape (batch_size, response_size)
        - rewards: Tensor of shape (batch_size, response_size)

        Output:
        - advantages: Tensor of shape (batch_size, response_size)
        - returns: Tensor of shape (batch_size, response_size)
        """
        if isinstance(values, list):
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards

        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns(
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """

        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            returns = []
            for r in rewards:
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        if action_mask is not None:
            rewards = action_mask * rewards

        # Calculate returns by accumulating discounted rewards
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return

        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines
        self.packing_samples = packing_samples

        if self.custom_reward_func:
            self.custom_reward_func = ray.remote(self.custom_reward_func)

    @torch.no_grad()
    def make_experience_list(self, extra_rm_args, all_prompts: Union[Dict, List[Dict]], **generate_kwargs) -> List[Experience]:
        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }
        min_samples_per_prompt = self.strategy.args.min_samples_per_prompt
        if min_samples_per_prompt == 0:
            experiences = super().make_experience_list(extra_rm_args, all_prompts, **generate_kwargs)
        else:
            experiences = self.make_experience_list_iterative(extra_rm_args, all_prompts, **generate_kwargs)

        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu)
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[Dict], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        return self._generate_vllm(all_prompts, **generate_kwargs)

    @torch.no_grad()
    def make_experience(self, extra_rm_args, samples: Samples) -> Experience:
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        self.actor.eval()
        device = torch.cuda.current_device()

        # extract values from samples
        sequences = samples.sequences
        attention_mask = samples.attention_mask
        action_mask = samples.action_mask
        num_actions = samples.num_actions
        packed_seq_lens = samples.packed_seq_lens

        start = time.time()
        sequences_cpu, attention_mask_cpu = (
            sequences.to("cpu"),
            attention_mask.to("cpu"),
        )

        # init log probs
        if self.initial_model is not None:
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            ) if self.initial_model is not None else None

        # values
        if self.critic is not None:
            value_ref = self.critic.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )
            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward:
                ray.get([value_ref])
                ray.get([self.critic.empty_cache.remote()])
        else:
            value_ref = ray.put(None)

        if self.strategy.args.colocate_actor_ref and (self.initial_model is not None):
            ray.get([base_action_log_probs_ref])
            ray.get([self.initial_model.empty_cache.remote()])

        # rewards
        r_refs = []
        # support remote RM API with ray
        if not self.remote_rm_url:
            for rm in self.reward_model:
                r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
        else:
            # remote RM
            if not self.packing_samples:
                queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
            else:
                sequences_list = []
                offset = 0
                tokens_list = sequences_cpu.tolist()[0]
                for length in packed_seq_lens:
                    sequences_list.append(tokens_list[offset : offset + length])
                    offset += length
                queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)

            if self.custom_reward_func:
                r = self.custom_reward_func.remote(queries, samples.prompts, samples.prompt_metadata, **extra_rm_args)
                r_refs.append(r)
            else:
                for rm in self.remote_rm_url:
                    r = remote_rm_fn_ray.remote(rm, queries=queries, prompts=samples.prompts)
                    r_refs.append(r)

        # log probs
        action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)
        actor_value_rm_time = time.time() - start

        # wait initial/critic/reward model done
        start = time.time()
        if self.initial_model is not None:
            ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
        else:
            ref_values = ray.get([value_ref] + r_refs)
        wait_time = time.time() - start

        if self.initial_model is not None:
            base_action_log_probs, value, rewards = ref_values[0], ref_values[1], ref_values[2:]
            base_action_log_probs = base_action_log_probs.to(device)
        else:
            value, rewards = ref_values[0], ref_values[1:]
        if value is not None:
            value = value.to(device)

        reward_metrics = {}
        if (len(rewards) > 0) and (type(rewards[0]) == dict):
            for k, v in rewards[0].items():
                if not k.startswith('metric_'):
                    continue
                reward_metrics[k] = [r[k] for r in rewards]
                reward_metrics[k] = self.reward_fn(reward_metrics[k])

        rewards = [(r['reward'] if type(r) == dict else r).to(device) for r in rewards]
        r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0]

        # avoid CUDA OOM when colocate models
        if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
            ray.get([self.reward_model[0].empty_cache.remote()])

        if self.strategy.args.colocate_actor_ref:
            torch.cuda.empty_cache()

        if self.initial_model is not None:
            kl = compute_approx_kl(
                action_log_probs,
                base_action_log_probs,
                action_mask=action_mask,
                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
            )
        else:
            kl = torch.zeros_like(action_log_probs)

        if not self.packing_samples:
            kl_mean = masked_mean(kl, action_mask, dim=-1)
        else:
            # convert tensor into list of tensors so that it's easier to manipulate
            # within dataset.
            sequences = unpacking_samples(sequences, packed_seq_lens)
            attention_mask = None
            action_log_probs = unpacking_samples(action_log_probs, num_actions)
            if value is not None:
                value = unpacking_samples(value, num_actions)

            kl = unpacking_samples(kl, num_actions)
            kl_mean = torch.tensor([each_kl.mean() for each_kl in kl], device=device)

        info = {
            "kl": kl_mean,
            "reward": r,
            "response_length": samples.response_length,
            "total_length": samples.total_length,
            "num_actions": num_actions,
        }
        info.update(reward_metrics)

        if self.strategy.args.perf:
            self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
            self.perf_stats["wait_time"] += wait_time

        experience = Experience(
            sequences,
            action_log_probs,
            value,
            None,
            None,
            attention_mask,
            action_mask,
            info,
            kl,
        )

        self.actor.train()  # reset model state
        return experience

    def _generate_vllm(self, all_prompts: List[Dict], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        all_prompts, all_prompt_metadata = all_prompts
        # round-robin load balance
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # Expand prompt list based on the number of samples per prompt
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        all_prompt_metadata = sum([[deepcopy(prompt_metadata) for _ in range(args.n_samples_per_prompt)] for prompt_metadata in all_prompt_metadata], [])
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms)
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]
            if prompt_token_ids:
                all_output_refs.append(
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])

        samples_list = []
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]
            prompts = all_prompts[i : i + self.strategy.args.micro_rollout_batch_size]
            cur_metadata = all_prompt_metadata[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0
                for output in outputs:
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                for output in outputs:
                    # left padding input
                    input_len = len(output.prompt_token_ids)
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)

                    # right padding output
                    output_len = len(output.outputs[0].token_ids)
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)

                    # concat input and output
                    sequences.append(input_ids + output_ids)

                sequences = torch.tensor(sequences)
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=action_mask,
                        num_actions=action_mask.size(1),
                        packed_seq_lens=None,
                        response_length=action_mask.float().sum(dim=-1),
                        total_length=attention_mask.float().sum(dim=-1),
                        prompts=prompts,
                        prompt_metadata=cur_metadata,
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []
                packed_seq_lens = []
                attention_mask = []
                num_actions = []
                for i, output in enumerate(outputs):
                    input_len = len(output.prompt_token_ids)
                    output_len = len(output.outputs[0].token_ids)
                    packed_seq_lens.append(input_len + output_len)
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                    attention_mask.extend([i + 1] * (input_len + output_len))

                    # current_action_mask = [0] * (input_len - 1) + [1] * output_len + [0]
                    # num_actions.append(max(1, sum(current_action_mask)))
                    num_actions.append(max(1, output_len))

                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        prompts=prompts,
                        prompt_metadata=cur_metadata,
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None


    def _iterative_generate_vllm(self, all_prompts: List[Dict], **kwargs) -> List[Experience]:
        """
        An iterative version of vLLM generation that uses min_samples_per_prompt,
        doubles them each iteration, calls self.make_experience to get reward,
        and uses the "metric_pass@1" field as a binary pass/fail for continuing.
        Returns a list of Experience objects directly (rather than Samples).
        """
        from vllm import SamplingParams

        # CHANGED: We'll parse out min_samples_per_prompt, etc.
        min_samples_per_prompt = kwargs.get("min_samples_per_prompt", 1)
        args = self.strategy.args
        n_samples_per_prompt = args.n_samples_per_prompt

        all_prompts, all_prompt_metadata = all_prompts
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
        )

        # CHANGED: iterative approach
        total_generated_for_prompt = [0] * len(all_prompts)
        P = list(range(len(all_prompts)))  # active prompt indices
        iteration = 0

        # We'll store a final list of Experience objects
        all_experiences = []

        while P:
            # how many has the "worst" prompt in P generated so far?
            max_generated_for_active = max(total_generated_for_prompt[i] for i in P)

            # current iteration's samples per prompt
            samples_per_prompt = max(
                0,
                min(
                    n_samples_per_prompt - max_generated_for_active,
                    min_samples_per_prompt * (2 ** iteration),
                ),
            )
            if samples_per_prompt <= 0:
                break

            # expand prompts for this iteration
            iteration_prompts = []
            iteration_metadata = []
            prompt_indices_for_this_round = []
            for i_idx in P:
                can_do = n_samples_per_prompt - total_generated_for_prompt[i_idx]
                if can_do > 0:
                    to_do = min(can_do, samples_per_prompt)
                    iteration_prompts.extend([all_prompts[i_idx]] * to_do)
                    iteration_metadata.extend([deepcopy(all_prompt_metadata[i_idx])] * to_do)
                    prompt_indices_for_this_round.extend([i_idx] * to_do)

            if not iteration_prompts:
                break

            # Tokenize for vLLM
            iteration_token_ids = self.tokenize_fn(iteration_prompts, self.prompt_max_len, padding=False)["input_ids"]

            # round-robin distribution to our engines
            batch_size = (len(iteration_token_ids) + len(llms) - 1) // len(llms)
            all_output_refs = []
            for e_i, llm in enumerate(llms):
                sub_ids = iteration_token_ids[e_i * batch_size : (e_i + 1) * batch_size]
                if sub_ids:
                    all_output_refs.append(
                        llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=sub_ids)
                    )

            # gather results
            all_outputs = sum(ray.get(all_output_refs), [])

            # Now we build "Samples" in micro-batches, but immediately convert to Experience
            # by calling self.make_experience(...) to get the reward.
            micro_experiences = []
            out_index = 0
            for i_out in range(0, len(all_outputs), args.micro_rollout_batch_size):
                outputs = all_outputs[i_out : i_out + args.micro_rollout_batch_size]
                cur_prompts = iteration_prompts[i_out : i_out + args.micro_rollout_batch_size]
                cur_metadata = iteration_metadata[i_out : i_out + args.micro_rollout_batch_size]

                # Build a single Samples object from these outputs (same logic as original),
                # then pass it to make_experience
                if not self.packing_samples:
                    max_input_len, max_output_len = 0, 0
                    for output in outputs:
                        max_input_len = max(max_input_len, len(output.prompt_token_ids))
                        max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                    pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                    sequences_list = []
                    for output in outputs:
                        input_ids = list(output.prompt_token_ids)
                        output_ids = list(output.outputs[0].token_ids)
                        # pad left for input
                        if len(input_ids) < max_input_len:
                            input_ids = [pad_token_id] * (max_input_len - len(input_ids)) + input_ids
                        # pad right for output
                        if len(output_ids) < max_output_len:
                            output_ids = output_ids + [pad_token_id] * (max_output_len - len(output_ids))
                        sequences_list.append(input_ids + output_ids)

                    sequences_tensor = torch.tensor(sequences_list)
                    (sequences_tensor, attn_mask, act_mask) = self.actor.process_sequences(
                        sequences_tensor, max_input_len, eos_token_id, pad_token_id
                    )

                    samples_obj = Samples(
                        sequences=sequences_tensor.to("cuda"),
                        attention_mask=attn_mask.to("cuda"),
                        action_mask=act_mask.to("cuda"),
                        num_actions=act_mask.size(1),
                        packed_seq_lens=None,
                        response_length=act_mask.float().sum(dim=-1),
                        total_length=attn_mask.float().sum(dim=-1),
                        prompts=cur_prompts,
                        prompt_metadata=cur_metadata,
                    )
                else:
                    # packing path (not fully tested in the example)
                    pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                    seqs = []
                    lens = []
                    attention_ids = []
                    num_actions = []
                    for idx2, output in enumerate(outputs):
                        in_len = len(output.prompt_token_ids)
                        out_len = len(output.outputs[0].token_ids)
                        seqs.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))
                        lens.append(in_len + out_len)
                        attention_ids.extend([idx2 + 1] * (in_len + out_len))
                        num_actions.append(max(1, out_len))

                    sequences_tensor = torch.tensor(seqs, device="cuda").unsqueeze(0)
                    attention_mask_tensor = torch.tensor(attention_ids, device="cuda").unsqueeze(0)
                    response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                    total_length = torch.tensor(lens, device="cuda", dtype=torch.float)

                    samples_obj = Samples(
                        sequences=sequences_tensor,
                        attention_mask=attention_mask_tensor,
                        action_mask=None,
                        num_actions=num_actions,
                        packed_seq_lens=torch.tensor(lens, device="cuda"),
                        response_length=response_length,
                        total_length=total_length,
                        prompts=cur_prompts,
                        prompt_metadata=cur_metadata,
                    )

                # Now call make_experience to get the reward & Experience
                exp = self.make_experience(kwargs.get("extra_rm_args", {}), samples_obj)
                micro_experiences.append(exp)

            # We now have a list of Experience objects (micro_experiences).
            # We must check which prompts had zero reward *across all newly generated samples*.
            # Collect rewards in the same index order as prompt_indices_for_this_round.
            # The iteration outputs are in the same order as iteration_prompts, so we match them 1:1.

            # Flatten experiences to get reward in the same order
            # (each micro-batch in micro_experiences is a chunk of them).
            iteration_exps_flat = []
            for me in micro_experiences:
                iteration_exps_flat.append(me)
            # Each Experience has a batch dimension in sequences, so we can match them 1-by-1 or do one big batch.
            # For simplicity, we do it 1-by-1, assuming micro-batch sizes are small.

            # We need to break them into single elements if the micro-batch size > 1.
            # In the naive code, each "Experience" can be a batch. If batch_size>1, we have multiple rows of reward.
            # We'll handle them row by row:
            expanded_exps = []
            for e in iteration_exps_flat:
                seqs = e.sequences
                if seqs.dim() == 2:
                    batch_sz = seqs.size(0)
                    # slice row-by-row
                    for row_i in range(batch_sz):
                        single_exp = Experience(
                            sequences=seqs[row_i : row_i + 1],
                            action_log_probs=e.action_log_probs[row_i : row_i + 1] if e.action_log_probs is not None else None,
                            values=e.values[row_i : row_i + 1] if e.values is not None else None,
                            returns=None,
                            advantages=None,
                            attention_mask=e.attention_mask[row_i : row_i + 1] if e.attention_mask is not None else None,
                            action_mask=e.action_mask[row_i : row_i + 1] if e.action_mask is not None else None,
                            info={k: (v[row_i] if isinstance(v, torch.Tensor) and v.dim() > 0 else v)
                                  for k, v in e.info.items()},
                            kl=e.kl[row_i : row_i + 1] if e.kl is not None else None,
                        )
                        expanded_exps.append(single_exp)
                else:
                    expanded_exps.append(e)

            # Now expanded_exps should line up exactly with iteration_prompts in order
            # We'll accumulate them in all_experiences
            all_experiences.extend(expanded_exps)

            # Summaries of reward 0 or not
            # We find the chunk of expanded_exps that belongs to each prompt in this iteration
            # Then sum their rewards. If sum is 0, keep the prompt in P.
            # Otherwise remove it from P.

            idx_offset = 0
            new_P = []
            for i_idx in P:
                can_do = n_samples_per_prompt - total_generated_for_prompt[i_idx]
                # how many we actually generated for i_idx
                to_do = min(can_do, samples_per_prompt)
                if to_do <= 0:
                    # might have been 0 if n_samples_per_prompt was already reached
                    continue

                # rewards for these 'to_do' expansions
                sub_exps = expanded_exps[idx_offset : idx_offset + to_do]
                idx_offset += to_do

                # Each sub_exp has info["reward"], which is a 1D shape with 1 element (we forced row-wise).
                sum_reward = sum(e.info["metric_pass@1"].item() for e in sub_exps)
                # update the total gen
                total_generated_for_prompt[i_idx] += to_do

                # if sum_reward == 0, we keep going if we haven't hit the max
                if (sum_reward == 0) and (total_generated_for_prompt[i_idx] < n_samples_per_prompt):
                    new_P.append(i_idx)

            P = new_P
            iteration += 1

        # Return all_experiences from all iterations
        return all_experiences

    # ADDED: a new specialized method that does advantage calculation after the iterative experiences
    @torch.no_grad()
    def make_experience_list_iterative(self, extra_rm_args, all_prompts: Union[Dict, List[Dict]], **generate_kwargs) -> List[Experience]:
        """
        This is the new pipeline for vLLM iterative generation. We call our replaced
        _iterative_generate_vllm (which returns Experience objects directly, each with reward already set).
        Then we do advantage calculations and return them.
        """
        # ADDED: ensure we are indeed in vLLM mode
        if self.vllm_engines is None:
            raise AssertionError("make_experience_list_iterative called but vllm_engines is None!")

        # ADDED: pass extra_rm_args into _generate_vllm
        generate_kwargs["extra_rm_args"] = extra_rm_args

        # Step 1: do iterative generation, collecting Experience objects
        experiences = self._iterative_generate_vllm(all_prompts, **generate_kwargs)

        # Step 2: process experiences (maybe RLOO or normal)
        experiences, rewards = self.process_experiences(experiences)

        # Step 3: advantage calc
        args = self.strategy.args
        for experience, reward in zip(experiences, rewards):
            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
            )

            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce", "rloo"]:
                experience.returns = self.get_cumulative_returns(
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                )
                experience.advantages = deepcopy(experience.returns)
            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")

            # calculate the return info.
            if not getattr(self, "packing_samples", False):
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
            experience.info["return"] = return_sums
            # remove unnecessary info
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")

        return experiences
