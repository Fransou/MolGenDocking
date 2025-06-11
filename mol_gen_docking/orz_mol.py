"""
Qwen2.5-0.5B base model + ppo

running command in 1 nodes:
directly run `python -m playground.orz_0p5b_ppo` is fine
"""

import asyncio
import copy
import json
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ray
import torch
import wandb
from loguru import logger
from omegaconf.listconfig import ListConfig
from orz.exps.examples.ppo.ppo_base_exp import BasePPOExp, BasePPOExpConfig
from orz.ppo import RayPPOTrainer
from orz.ppo.utils import check_reflection_pattern
from rdkit import Chem, RDLogger
from typing_extensions import override

from mol_gen_docking.playground.zero_setting_base import (
    CustomDataset,
    EvalCustomDataset,
)
from mol_gen_docking.reward.rl_rewards import RewardScorer

# set wandb offline
os.environ["WANDB_MODE"] = "offline"
os.environ["VLLM_USE_V1"] = "0"
RDLogger.DisableLog("rdApp.*")


DEBUG_MODE = (
    False if os.environ.get("DEBUG_MODE", "False") == "False" else True
)  # Global debug flag

file_name = (
    f"{'debug_' if DEBUG_MODE else ''}{os.path.splitext(os.path.basename(__file__))[0]}"
)
executor = ThreadPoolExecutor(max_workers=64)


@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True
    use_orm_score: bool = False

    # Conditional settings with production values first
    num_gpus_per_node: int = 4
    num_nodes: int = 1

    # resource related settings
    scorer_ncpus: int = 4
    scorer_exhaustivness: int = 1  # TODO change to 4

    ref_num_nodes: int = num_nodes * (num_gpus_per_node // 4)
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = num_nodes * (num_gpus_per_node // 4)
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = num_nodes * (num_gpus_per_node // 4)
    critic_num_gpus_per_node: int = 1
    reward_num_nodes: int = num_nodes * (num_gpus_per_node // 4)
    reward_num_gpus_per_node: int = 1
    colocate_all: bool = DEBUG_MODE
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = (
        num_nodes * (2 * num_gpus_per_node // 4) if not DEBUG_MODE else 1
    )
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = "SFT_SMOL/model"
    reward_pretrain: Optional[str] = None
    save_interval: int = 50
    ckpt_path: str = f"/scratch/fransou/orz_ckpt/{file_name}"
    save_path: str = f"/scratch/fransou/orz_ckpt/{file_name}"
    tensorboard_log_dir: str = f"orz_logs/{file_name}"

    # MathTrain dataset and Math500 eval dataset
    # data related settings
    base_data_path: str = os.environ["ORZ_DATA_PATH"]
    n_prompts: int = 256
    prompt_data: ListConfig = ListConfig(
        [
            base_data_path + "/train_prompts.json",
        ]
    )
    eval_prompt_data: ListConfig = ListConfig(
        [base_data_path + "/eval_data/eval_prompts.json"]
    )
    prompt_data_probs: ListConfig = ListConfig([1.0])

    # ppo related settings
    actor_learning_rate: float = 1e-6
    critic_learning_rate: float = 5e-6
    num_warmup_steps: int = 50
    prompt_max_len: int = 512
    enable_prefix_caching: bool = True
    update_ref_every_epoch: bool = True
    advantage_normalize: bool = True

    num_episodes: int = 20
    rollout_batch_size: int = 128 if not DEBUG_MODE else 4
    n_samples_per_prompt: int = 64 if not DEBUG_MODE else 2
    micro_rollout_batch_size: int = 128 if not DEBUG_MODE else 4

    policy_update_steps: int = 1
    critic_update_steps: int = 12 if not DEBUG_MODE else 1
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = 0.0
    use_kl_loss: bool = True
    use_kl_estimator_k3: bool = True

    enable_eval: bool = True
    eval_interval: int = 10

    # generate related settings
    generate_max_len: int = 512  # change to larger later
    max_len: int = 512  # change to larger later
    packing_max_len: int = generate_max_len + max_len

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop: ListConfig = ListConfig(["User:", "Human:", "Assistant:", "</answer>"])

    # grpo related settings
    use_grpo: bool = False

    gpu_memory_utilization: float = 0.9 if not DEBUG_MODE else 0.5
    critic_pretrain: Optional[str] = "" if use_grpo else pretrain

    gamma: float = 1.0
    lambd: float = 1.0


class WandbWriter:
    def __init__(self, cfg: PPOExpConfig):
        if isinstance(cfg.pretrain, str):
            project_name = ("MolGenDocking_" + cfg.pretrain).replace("/", "-")
        else:
            project_name = "MolGenDocking"
        self.tables: Dict[str, pd.DataFrame] = {}
        wandb.init(project=project_name, name=cfg.ckpt_path, config=cfg)

    def add_scalar(self, key: str, value: float, step: int):
        wandb.log({key: value}, step=step)

    def add_dict(self, dict_vals: Dict[str, Any], step: int):
        wandb.log(dict, step=step)

    def add_histogram(self, key: str, value: np.ndarray, step: int):
        wandb.log({key: wandb.Histogram(value)}, step=step)

    def add_text(
        self,
        key: str,
        value: str,
        step: int,
    ):
        if key not in self.tables:
            cols = ["step", "text"]
            self.tables[key] = pd.DataFrame(columns=cols)

        self.tables[key].loc[self.tables[key].shape[0]] = [step, value]
        wandb.log({key: wandb.Table(dataframe=self.tables[key])})

    def update_table(self, table_name, step, data):
        if not step % 10 == 0:
            pass
        if table_name not in self.tables:
            cols = ["step"] + list(data.keys())
            self.tables[table_name] = pd.DataFrame(columns=cols)

        self.tables[table_name].loc[self.tables[table_name].shape[0]] = [step] + list(
            data.values()
        )

    def upload_table(self, table_name):
        if self.tables[table_name].step.max() % 50 == 0:
            wandb.log({table_name: wandb.Table(dataframe=self.tables[table_name])})

    def flush(self):
        pass

    def close(self):
        wandb.finish()


class CustomRewardTrainer(RayPPOTrainer):
    def __init__(self, cfg: PPOExpConfig, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.writer = WandbWriter(cfg)

        # Find the directory of the training set:
        data_path = os.path.dirname(cfg.prompt_data[0])

        self._reward_properties = (
            ray.remote(RewardScorer)
            .options(num_cpus=1)
            .remote(
                path_to_mappings=data_path,
                parse_whole_completion=True,
                oracle_kwargs=dict(
                    exhaustiveness=cfg.scorer_exhaustivness,
                    ncpu=cfg.scorer_ncpus,
                ),
            )  # type: ignore
        )

        self._reward_valid_smiles = self._reward_properties = (
            ray.remote(RewardScorer)
            .options(num_cpus=1)
            .remote(
                path_to_mappings=data_path,
                reward="valid_smiles",
                parse_whole_completion=True,
            )  # type: ignore
        )

    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        extras: List[dict],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        # make log metrics
        scores: List[float] = []
        responses: List[str] = []
        avg_non_stop_count = 0
        pass_at_n_dict: Dict[str, List[float]] = defaultdict(list)
        num_tokens: List[int] = []

        def get_mol_prop_score(p, res):
            return self._reward_properties.get_score.remote(p, res)

        def get_mol_valid_score(p, res):
            return self._reward_valid_smiles.get_score.remote(p, res)

        @ray.remote(num_cpus=1)
        def get_reflection_pattern_score(res):
            reflection_pattern_dict = check_reflection_pattern(res)
            reflection_pattern_num = sum(reflection_pattern_dict.values())
            return reflection_pattern_num

        rep_tasks = []
        responses = []
        for output in outputs:
            response = output["response"]
            # calculate repeat score for log
            responses.append(response)
            rep_tasks.append(get_reflection_pattern_score.remote(response))
        mol_rewards = get_mol_prop_score(prompts, responses)
        valid_reward = get_mol_valid_score(prompts, responses)

        reflection_pattern_scores = ray.get(rep_tasks)
        mol_scores = ray.get(mol_rewards)
        valid_scores = ray.get(valid_reward)

        for output in outputs:
            responses.append(output["response"])
        output_tokens = self._tokenize(
            responses, self.cfg.generate_max_len, padding=False
        )["input_ids"]

        n_logs = min(len(prompts), 128)
        for i in range(n_logs):
            self.writer.update_table(
                "generations",
                self.global_step,
                data={
                    "prompts": prompts[i],
                    "outputs": outputs[i]["response"],
                    "final_answer": outputs[i]["final_answer"],
                    "stop_reason": outputs[i]["stop_reason"],
                    "response_token": len(output_tokens[i]),
                    "mol_score": mol_scores[i],
                    "valid_score": valid_scores[i],
                },
            )
        self.writer.upload_table("generations")
        for idx in range(len(outputs)):
            prompt, output, out_token = prompts[idx], outputs[idx], output_tokens[idx]
            m_score, v_score, reflection_pattern_score = (
                mol_scores[idx],
                valid_scores[idx],
                reflection_pattern_scores[idx],
            )
            stop_reason = output["stop_reason"]
            response_token = len(out_token)
            output["molecule_score"] = float(m_score)
            output["valid_score"] = float(v_score)
            output["reflection_pattern_score"] = reflection_pattern_score
            # only correct and stoped response can aquire reward
            if stop_reason == "stop":
                score = m_score + v_score
            else:
                avg_non_stop_count += 1
                score = 0.0
            scores.append(score)

            # calculate pass@n
            pass_at_n_dict[prompt].append(scores[-1])
            # log num_tokens
            num_tokens.append(response_token)

        # must before grpo, for grpo will change scores
        num_tokens_arr = np.array(
            num_tokens, dtype=np.float32
        )  # must be float to calculate mean and std
        scores_arr = np.array(scores)
        correct_tokens_arr = (
            np.array([])
            if np.all(scores_arr == 0)
            else np.array(num_tokens_arr[scores_arr == 1])
        )
        incorrect_tokens_arr = (
            np.array([])
            if np.all(scores_arr == 1)
            else np.array(num_tokens_arr[scores_arr == 0])
        )

        # GRPO
        if self.cfg.use_grpo:
            self.writer.add_scalar("grpo_raw_reward", np.mean(scores), self.global_step)
            # grpo reward normalization
            for i, prompt in enumerate(prompts):
                scores[i] -= float(np.mean(pass_at_n_dict[prompt]))
                if std := np.std(pass_at_n_dict[prompt]) > 0:
                    scores[i] /= std

        def dump_results(prompts, outputs, scores):
            saved = []
            for prompt, output, score in zip(prompts, outputs, scores):
                if isinstance(score, torch.Tensor):
                    score = score.tolist()
                if isinstance(output, torch.Tensor):
                    output = output.tolist()
                if isinstance(prompt, torch.Tensor):
                    prompt = prompt.tolist()
                saved.append(dict(prompt=prompt, score=score, outputs=output))
            json.dump(
                saved,
                open(
                    os.path.join(
                        self.cfg.save_path,
                        f"iter{self.global_step}_generation_results.json",
                    ),
                    "w",
                ),
                ensure_ascii=False,
                indent=2,
            )

        global executor
        asyncio.get_event_loop().run_in_executor(
            executor,
            dump_results,
            copy.deepcopy(prompts),
            copy.deepcopy(outputs),
            copy.deepcopy(scores),
        )

        log_dict = {
            "avg_non_stop_count": avg_non_stop_count / len(prompts),
            "avg_molecular_score": sum(mol_scores) / len(prompts),
            "avg_valid_score": sum(valid_scores) / len(prompts),
            "avg_reflection_pattern_score": sum(reflection_pattern_scores)
            / len(prompts),
            "avg_pass_at_n": np.mean([np.max(v) for v in pass_at_n_dict.values()])
            / len(pass_at_n_dict),
            "avg_num_tokens": np.mean(num_tokens_arr).item(),
            "std_num_tokens": np.std(num_tokens_arr).item(),
            "avg_correct_num_tokens": 0
            if len(correct_tokens_arr) == 0
            else np.mean(correct_tokens_arr).item(),
            "std_correct_num_tokens": 0
            if len(correct_tokens_arr) == 0
            else np.std(correct_tokens_arr).item(),
            "avg_incorrect_num_tokens": 0
            if len(incorrect_tokens_arr) == 0
            else np.mean(incorrect_tokens_arr).item(),
            "std_incorrect_num_tokens": 0
            if len(incorrect_tokens_arr) == 0
            else np.std(incorrect_tokens_arr).item(),
        }
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.global_step)
        logging_str = ",".join([f"{k}: {v:.4f}" for k, v in log_dict.items()])
        logger.info(logging_str)

        # make histogram for correct and incorrect response length
        if len(correct_tokens_arr) > 0:
            self.writer.add_histogram(
                "correct_response_length", correct_tokens_arr, self.global_step
            )
        if len(incorrect_tokens_arr) > 0:
            self.writer.add_histogram(
                "incorrect_response_length", incorrect_tokens_arr, self.global_step
            )

        # make a pre-token score tensor for each output, for example: [0, 0, 0, 0, r]
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # rm empty response
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)
        return res_prompts, res_responses, res_score_tensors

    @torch.no_grad()
    async def generate_vllm(
        self,
        gen_func: Callable[[Any], Awaitable[List[str | Any]]],
        prompts: List[str],
        extras: List[dict],
        **kwargs,
    ) -> List[dict[str, Any]]:
        from vllm import SamplingParams

        # read sampling params from self.cfg

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            top_k=self.cfg.top_k,
            max_tokens=self.cfg.generate_max_len,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
            stop=self.cfg.stop,
        )
        responses, stop_reasons = await gen_func(  # type: ignore
            prompts=prompts,  # type: ignore
            sampling_params=sampling_params,  # type: ignore
            use_tqdm=False,  # type: ignore
            truncate_prompt=True,  # type: ignore
        )

        @ray.remote(num_cpus=1)
        def extract_final_answers_batch(responses: List[str]) -> List[Any | str]:
            # pattern = re.compile(r"(\\boxed{.*})")
            RDLogger.DisableLog("rdApp.*")

            final_answers = []
            for comp in responses:
                re.split("\n| |.", comp)
                # Then we filter by removing any string that does not contain "C"
                s_poss = [
                    x
                    for x in comp.split()
                    if ("C" in x or x.count("c") > 1) and "e" not in x
                ]
                # Finally we remove any string that is not a valid SMILES
                s_spl = [x + " - " for x in s_poss if Chem.MolFromSmiles(x) is not None]

                final_answers.append("".join(s_spl))
            return final_answers

        BATCH_SIZE = 16
        num_batches = (len(responses) + BATCH_SIZE - 1) // BATCH_SIZE
        extract_tasks = []
        for i in range(num_batches):
            start_idx = i * BATCH_SIZE
            end_idx = min((i + 1) * BATCH_SIZE, len(responses))
            batch = responses[start_idx:end_idx]
            extract_tasks.append(extract_final_answers_batch.remote(batch))  # type: ignore
        batched_results = await asyncio.gather(
            *[asyncio.to_thread(ray.get, task) for task in extract_tasks]
        )
        final_answers = [answer for batch in batched_results for answer in batch]

        results = []
        for extra, response, stop_reason, final_answer in zip(
            extras, responses, stop_reasons, final_answers
        ):
            results.append(
                dict(
                    response=response,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                )
            )

        return results

    @override
    async def eval(self):
        logger.info("Start evaluating on val set")
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.generate_max_len,
            stop=self.cfg.stop,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )

        from torch.utils.data import DataLoader

        dataset = self.eval_dataset
        dataloader = DataLoader(
            dataset, batch_size=len(dataset), shuffle=False, drop_last=False
        )
        prompt_pre_llm = (
            len(dataset) + self.cfg.vllm_num_engines - 1
        ) // self.cfg.vllm_num_engines

        output_for_save = []
        log_dict = defaultdict(float)
        for batch in dataloader:
            prompts = list(batch[0])
            answers = list(batch[1]["answer"])
            file_names = list(batch[1].get("file_name"))
            outputs = []
            for i, llm in enumerate(self.vllm_engines):
                outputs.append(
                    llm.generate.remote(
                        prompts=prompts[i * prompt_pre_llm : (i + 1) * prompt_pre_llm],
                        sampling_params=sampling_params,
                    )
                )
            outputs = await asyncio.gather(*outputs)
            outputs = sum(outputs, [])

            final_answers = []
            pattern = re.compile(r"<answer>.*?</answer>", re.DOTALL)  # TODO
            for output in outputs:
                matches = re.findall(pattern, output.outputs[0].text)
                if len(matches) > 0:
                    final_answers.append(matches[-1])
                else:
                    final_answers.append("")

            for prompt, output, final_answer, answer, file_name in zip(
                prompts, outputs, final_answers, answers, file_names
            ):
                output_for_save.append(
                    dict(
                        prompt=prompt,
                        output=output.outputs[0].text,
                        final_answer=final_answer,
                        answer=answer,
                    )
                )
                log_dict[f"{file_name}/total_response_len_in_char"] += len(
                    output.outputs[0].text
                )
                log_dict[f"{file_name}/total"] += 1
        # get all file_names from self.cfg.eval_prompt_data
        all_file_names: List[str] = [
            os.path.splitext(os.path.basename(file_path))[0]
            for file_path in self.cfg.eval_prompt_data
        ]

        for file_name in all_file_names:
            try:
                log_dict[f"{file_name}/response_len_in_char"] = (
                    log_dict[f"{file_name}/total_response_len_in_char"]
                    / log_dict[f"{file_name}/total"]
                )
            except Exception as e:
                print(e)
                print("==================================")
            log_dict.pop(f"{file_name}/total_response_len_in_char")
            log_dict.pop(f"{file_name}/total")


class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        vllm_engines = self.create_inference_engine()
        return CustomRewardTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            vllm_engines=vllm_engines,
            colocate_pg=self.get_colocate_pg,
        )

    @override
    @cached_property
    def train_dataset(self):
        dialogues = []
        for file_path in self.cfg.prompt_data:
            with open(file_path) as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = CustomDataset(
            self.cfg.n_prompts,
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset

    @override
    @cached_property
    def eval_dataset(self):
        dialogues = []
        for file_path in self.cfg.eval_prompt_data:
            with open(file_path) as f:
                dialogues.extend(json.load(f))
        logger.info(f"Start processing {len(dialogues)} dialogues")
        prompts_dataset = EvalCustomDataset(
            dialogues,
            self.tokenizer,
            self.cfg.prompt_max_len,
            self.strategy,
            pretrain_mode=False,
            num_processors=1,
        )
        logger.info(f"Finished processing {len(prompts_dataset)} dialogues")
        return prompts_dataset


if __name__ == "__main__":
    cfg = PPOExpConfig()
    exp = PPOExp().set_cfg(cfg)
    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(exp.run())
