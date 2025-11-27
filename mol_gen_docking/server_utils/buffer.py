import asyncio
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np
import ray

from mol_gen_docking.server_utils.utils import (
    MolecularVerifierQuery,
    MolecularVerifierResponse,
)


class RewardBuffer:
    def __init__(
        self, app: Any, buffer_time: float = 1.0, max_batch_size: int = 16
    ) -> None:
        """
        Buffers /get_reward queries and processes them in merged batches.
        """
        self.app = app
        self.buffer_time = buffer_time
        self.max_batch_size = max_batch_size
        self.queue: deque[Tuple[MolecularVerifierQuery, asyncio.Future]] = deque()
        self.lock = asyncio.Lock()
        self.processing_task = asyncio.create_task(self._batch_loop())

    async def add_query(
        self, query: MolecularVerifierQuery
    ) -> MolecularVerifierResponse:
        """
        Adds a query to the buffer and waits for the corresponding result.
        """
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        async with self.lock:
            self.queue.append((query, future))
        return await future  # type:ignore

    async def _batch_loop(self) -> None:
        """
        Background task that continuously processes queries in buffered batches.
        """
        while True:
            await asyncio.sleep(self.buffer_time)
            await self._process_pending_queries()

    async def _process_pending_queries(self) -> None:
        async with self.lock:
            if not self.queue:
                return

            batch = [
                self.queue.popleft()
                for _ in range(min(len(self.queue), self.max_batch_size))
            ]

        queries, futures = zip(*batch)

        try:
            responses = await self._process_batch(list(queries))
            for fut, res in zip(futures, responses):
                if not fut.done():
                    fut.set_result(res)
        except Exception as e:
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

    async def _process_batch(
        self, queries: List[MolecularVerifierQuery]
    ) -> List[MolecularVerifierResponse]:
        """
        Processes a list of MolecularVerifierQuery in one Ray call.
        """
        app = self.app

        # --- Step 1. Merge all batched inputs ---
        all_completions: List[str] = []
        all_metadata: List[dict[str, Any]] = []
        query_indices: List[int] = []

        for i, q in enumerate(queries):
            all_completions.extend(q.query)
            all_metadata.extend(q.metadata)
            query_indices.extend([i] * len(q.query))

        if len(all_completions) == 0:
            # All failed early
            return [
                q
                if isinstance(q, MolecularVerifierResponse)
                else MolecularVerifierResponse(
                    reward=0.0, reward_list=[], error="Empty batch"
                )
                for q in queries
            ]

        # --- Step 2. Compute batched reward ---
        reward_actor = app.state.reward_model
        valid_scorer = app.state.reward_valid_smiles

        # Run in parallel
        rewards_job = reward_actor.get_score.remote(
            completions=all_completions, metadata=all_metadata
        )
        valid_reward = valid_scorer.get_score(
            completions=all_completions, metadata=all_metadata
        )
        final_smiles: List[str] = valid_scorer.get_all_completions_smiles(
            completions=all_completions
        )

        out: Tuple[List[float], List[Dict[str, Any]]] = ray.get(rewards_job)
        rewards, metadatas = out
        # --- Step 3. Group results by original query ---
        grouped_results: List[List[float]] = [[] for _ in range(len(queries))]
        grouped_meta: List[List[Dict[str, Any]]] = [[] for _ in range(len(queries))]
        grouped_smiles: List[List[str]] = [[] for _ in range(len(queries))]

        for r, m, s, idx in zip(rewards, metadatas, final_smiles, query_indices):
            grouped_results[idx].append(r)
            grouped_meta[idx].append(m)
            grouped_smiles[idx].append(s)

        # --- Step 4. Compute per-query metrics ---
        responses = []
        for i, q in enumerate(queries):
            if isinstance(q, MolecularVerifierResponse):
                # prefilled error
                responses.append(q)
                continue
            rewards_i = grouped_results[i]
            metadata_i = grouped_meta[i]
            smiles_i = grouped_smiles[i]
            metadata = q.metadata
            prompts = []
            for meta in metadata:
                assert all(k in meta for k in ["properties", "objectives", "target"])
                prompts.append(
                    "|".join(
                        [
                            f"{p}, {o}, {t}"
                            for p, o, t in zip(
                                meta["properties"], meta["objectives"], meta["target"]
                            )
                        ]
                    )
                )

            # diversity + uniqueness per query
            unique_prompts = list(set(prompts))
            group_prompt_smiles = {
                p: [
                    s[-1]
                    for s, p_ in zip(smiles_i, prompts)
                    if (p_ == p) and not s == []  # type: ignore
                ]
                for p in unique_prompts
            }

            diversity_scores_dict = {
                p: app.state.diversity_evaluator(group_prompt_smiles[p])
                if len(group_prompt_smiles[p]) > 1
                else 0
                for p in unique_prompts
            }
            diversity_score = [float(diversity_scores_dict[p]) for p in prompts]
            diversity_score = [d if not np.isnan(d) else 0 for d in diversity_score]

            uniqueness_scores_dict = {
                p: app.state.uniqueness_evaluator(group_prompt_smiles[p])
                if len(group_prompt_smiles[p]) > 1
                else 0
                for p in unique_prompts
            }
            uniqueness_score = [float(uniqueness_scores_dict[p]) for p in prompts]
            uniqueness_score = [u if not np.isnan(u) else 0 for u in uniqueness_score]

            max_per_prompt_dict = {
                p: max([float(r) for r, p_ in zip(rewards_i, prompts) if p_ == p])
                for p in unique_prompts
            }
            max_per_prompt = [max_per_prompt_dict[p] for p in prompts]

            response = MolecularVerifierResponse(
                reward=0.0 if len(rewards_i) == 0 else sum(rewards_i) / len(rewards_i),
                reward_list=rewards_i,
                meta={
                    "property_scores": rewards_i,
                    "validity": valid_reward,
                    "uniqueness": uniqueness_score,
                    "diversity": diversity_score,
                    "pass_at_n": max_per_prompt,
                    "rewards": rewards_i,
                    "verifier_metadata_output": metadata_i,
                },
                error=None,
            )
            responses.append(response)

        return responses
