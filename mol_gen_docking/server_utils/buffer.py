import asyncio
import logging
from collections import deque
from typing import Any, Dict, List, Tuple

import ray

from mol_gen_docking.server_utils.utils import (
    MolecularVerifierServerMetadata,
    MolecularVerifierServerQuery,
    MolecularVerifierServerResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("molecular_verifier_buffer")
logger.setLevel(logging.INFO)


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
        self.queue: deque[Tuple[MolecularVerifierServerQuery, asyncio.Future]] = deque()
        self.lock = asyncio.Lock()
        self.processing_task = asyncio.create_task(self._batch_loop())

    async def add_query(
        self, query: MolecularVerifierServerQuery
    ) -> MolecularVerifierServerResponse:
        """
        Adds a query to the buffer and waits for the corresponding result.
        """
        try:
            future: asyncio.Future = asyncio.get_event_loop().create_future()
            async with self.lock:
                self.queue.append((query, future))
            return await future  # type:ignore
        except Exception as e:
            logger.error(f"Error in add_query: {e}")
            raise e

    async def _batch_loop(self) -> None:
        """
        Background task that continuously processes queries in buffered batches.
        """
        while True:
            await asyncio.sleep(self.buffer_time)
            await self._process_pending_queries()

    async def _process_pending_queries(self) -> None:
        try:
            async with self.lock:
                if not self.queue:
                    return

                batch = [
                    self.queue.popleft()
                    for _ in range(min(len(self.queue), self.max_batch_size))
                ]

            queries, futures = zip(*batch)
            logger.info(f"Processing batch of size {len(queries)}")
            try:
                responses = await self._process_batch(list(queries))
                for fut, res in zip(futures, responses):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as e:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(e)
        except Exception as e:
            logger.error(f"Error in _process_pending_queries: {e}")
            raise e

    async def _process_batch(
        self, queries: List[MolecularVerifierServerQuery]
    ) -> List[MolecularVerifierServerResponse]:
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
                if isinstance(q, MolecularVerifierServerResponse)
                else MolecularVerifierServerResponse(
                    reward=0.0, reward_list=[], error="Empty batch"
                )
                for q in queries
            ]

        # --- Step 2. Compute batched reward ---
        reward_actor = app.state.reward_model

        # Run in parallel
        rewards_job = reward_actor.get_score.remote(
            completions=all_completions, metadata=all_metadata
        )

        out = ray.get(rewards_job)
        rewards = out.rewards
        metadatas = [m.model_dump() for m in out.verifier_metadatas]
        # --- Step 3. Group results by original query ---
        grouped_results: List[List[float]] = [[] for _ in range(len(queries))]
        grouped_meta: List[List[Dict[str, Any]]] = [[] for _ in range(len(queries))]

        for r, m, idx in zip(rewards, metadatas, query_indices):
            grouped_results[idx].append(r)
            grouped_meta[idx].append(m)

        # --- Step 4. Compute per-query metrics ---
        responses = []
        for i, q in enumerate(queries):
            if isinstance(q, MolecularVerifierServerResponse):
                # prefilled error
                responses.append(q)
                continue
            rewards_i = grouped_results[i]
            metadata_i = grouped_meta[i]
            response = MolecularVerifierServerResponse(
                reward=0.0 if len(rewards_i) == 0 else sum(rewards_i) / len(rewards_i),
                reward_list=rewards_i,
                meta=[MolecularVerifierServerMetadata(**m) for m in metadata_i],
                error=None,
            )
            responses.append(response)

        return responses
