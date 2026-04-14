# Prefix Caching Notes

## 1. What Prefix Caching Is

Prefix caching reuses KV cache blocks for requests that share the same prompt prefix. Its goal is to avoid recomputing prefill for the shared prefix, reduce TTFT, and improve throughput. It is a performance optimization only; model outputs should stay the same.

In this workspace, the **core algorithm** comes from upstream `vllm/`. `vllm-ascend/` mainly adapts it to:

- Ascend scheduler behavior
- DeepSeek MLA KV layout
- external KV connectors such as AscendStore, UCM, CPU offload, and Mooncake

Use this mental model:

- `vllm/`: source of truth for hash-based prefix caching
- `vllm-ascend/`: Ascend-specific scheduling, loading, and KV movement around that mechanism

## 2. Core Data Flow

The main objects are:

- `Request`: owns `prompt_token_ids`, `all_token_ids`, and `block_hashes`
- `KVCacheManager`: finds local cache hits, allocates blocks, caches full blocks, and frees blocks
- `BlockPool`: owns all KV blocks, free queue, ref counts, and cached-block hash map
- `KVConnector`: optionally provides external prefix hits beyond local cache

High-level flow:

1. A request is tokenized and wrapped as `Request`.
2. `Request.block_hashes` are generated for all full prompt blocks.
3. Scheduler asks `KVCacheManager` for local prefix hits.
4. Scheduler optionally asks `KVConnector` for external prefix hits.
5. Scheduler computes `num_computed_tokens = local_hits + external_hits`.
6. `allocate_slots()` attaches hit blocks and allocates only the uncached suffix.
7. Model runner computes the uncached suffix.
8. Newly completed full blocks are inserted into the prefix cache.
9. When the request finishes, blocks are freed; cached blocks may stay reusable.

## 3. Why `block_hashes` Form a Chain

Each full block hash depends on:

- parent block hash
- current block token IDs
- extra keys such as LoRA, multimodal hashes, `cache_salt`, and prompt-embed hashes

So the hash is not "this block alone"; it is "this block under this exact prefix". That prevents false reuse when two requests have identical current-block tokens but different earlier context.

Example with block size 4:

```text
Shared prefix:
Block0 -> Block1 -> Block2

Request A:
H0 = H(NONE, Block0, extra)
H1 = H(H0,   Block1, extra)
H2 = H(H1,   Block2, extra)
HA3 = H(H2,  SuffixA, extra)

Request B:
H0 = H(NONE, Block0, extra)
H1 = H(H0,   Block1, extra)
H2 = H(H1,   Block2, extra)
HB3 = H(H2,  SuffixB, extra)
```

The scheduler can therefore reuse `H0`, `H1`, and `H2`, then stop at the first miss.

Important details:

- only **full blocks** are hashed and cached
- a full prompt hit still recomputes the last token to obtain next-token logits
- `cache_salt` isolates the whole chain by changing the first block hash

## 4. Local Prefix Cache Lifecycle

### Lookup

The scheduler calls `get_computed_blocks()` to find the longest local cache hit. Matching is prefix-only and contiguous: once one block misses, lookup stops.

### Touch

Matched blocks are `touch`ed:

- `ref_cnt` is increased
- if a block was only cached and sitting in the free queue, it is removed from the free queue

This prevents the block from being evicted during the current scheduling step.

### Allocation

`allocate_slots()` combines five regions conceptually:

- `comp`: tokens already computed by this request
- `new_comp`: tokens newly hit from local prefix cache
- `ext_comp`: tokens newly hit from external cache
- `new`: tokens that must be computed now
- `lookahead`: speculative slots

The key effect is simple: the block table for a request becomes "reused prefix blocks + newly allocated suffix blocks".

### Cache Insert

When newly computed blocks become full, `cache_blocks()` inserts them into the hash map. Partial blocks are not cached.

### Free and Eviction

When a request finishes:

- block ref counts are decremented
- blocks with `ref_cnt == 0` go back to the free queue
- cached blocks remain reusable until later eviction

Eviction happens lazily when a free cached block is reallocated for another request.

### Reset

`reset_prefix_cache()` clears all cached hashes, but only if no normal blocks are still in use. It resets cache state, not model weights or KV tensor storage itself.

## 5. DeepSeek-V3.1 Walkthrough

DeepSeek-V3.1 in this repo runs through `DeepseekV3ForCausalLM` and uses the DeepSeek MLA path. On Ascend, MLA KV tensors are handled specially in the model runner, but prefix cache matching still follows the same hash/block logic described above.

Assume two requests share a long prompt prefix:

- Request A: shared prefix + question A
- Request B: shared prefix + question B

### Request A: cache warmup

1. Request A enters the engine and builds `block_hashes`.
2. Local prefix lookup returns zero hits because the cache is empty.
3. Scheduler allocates blocks for the full prompt.
4. DeepSeek-V3.1 MLA prefill computes the full prompt on NPU.
5. As full KV blocks become available, they are inserted into the prefix cache.
6. Request A decodes and finishes.
7. Its blocks are freed; cached blocks become reusable.

At this point, the shared prefix is now present in the prefix cache.

### Request B: actual prefix reuse

1. Request B enters and builds the same hash chain for the shared prefix.
2. Local lookup finds the longest matching prefix blocks.
3. Those blocks are `touch`ed and attached to Request B's block table.
4. Scheduler allocates only the suffix blocks for question B.
5. DeepSeek-V3.1 MLA prefill computes only the uncached suffix.
6. Newly completed suffix blocks are also inserted into the prefix cache.

So for DeepSeek-V3.1, prefix caching does **not** change the MLA math; it changes **which token span reaches MLA prefill at all**.

## 6. External Prefix Cache in `vllm-ascend`

Beyond local cache hits, Ascend scheduler may query a connector:

- `AscendStore` / KV Pool: standard hash-based external lookup using `request.block_hashes`
- `CPUOffloadingConnector`: treats CPU as a second-level prefix cache
- `UCMConnector`: external persistent prefix-cache backend
- `MooncakeConnector`: often used for remote prefill / remote decode transfer rather than plain hash lookup

The scheduler merges local and external hits into one number:

```text
num_computed_tokens = num_local_computed_tokens + num_external_computed_tokens
```

Then `allocate_slots()` treats both as already available prefix tokens.

This is why prefix caching in `vllm-ascend` is better thought of as a **multi-layer KV reuse system**, not just an in-device optimization.

## 7. Practical Limits and Caveats

- Prefix caching only helps when requests share long prefixes.
- Short or random prompts usually get little value.
- Only full blocks count; partial overlap does not help.
- Full prompt hits still re-run the last token.
- Some request types skip reading prefix cache entirely, such as prompt-logprob cases.
- DeepSeek-family MLA paths may have special KV layout handling in Ascend worker code, but the hit logic remains block-hash based.
- Hybrid KV-cache configurations can force larger alignment granularity, making effective cache-hit thresholds much larger.

## 8. How to Read the Code

Suggested reading order:

1. Upstream design: `vllm/docs/design/prefix_caching.md`
2. Request hash generation: `vllm/vllm/v1/request.py`
3. Block-hash helpers: `vllm/vllm/v1/core/kv_cache_utils.py`
4. Local cache manager: `vllm/vllm/v1/core/kv_cache_manager.py`
5. Block pool lifecycle: `vllm/vllm/v1/core/block_pool.py`
6. Ascend scheduler integration: `vllm-ascend/vllm_ascend/core/recompute_scheduler.py`
7. External KV integration: `vllm-ascend/vllm_ascend/distributed/kv_transfer/...`

## 9. One-Sentence Summary

Prefix caching in `vllm-ascend` works by turning a request's prompt into a chain of full-block hashes, reusing the longest matching prefix from local or external KV cache, and sending only the uncached suffix into actual DeepSeek-V3.1 MLA computation.

## Appendix: Source Code Index

Use this as a quick map when revisiting the code.

### Upstream `vllm/`

- `vllm/docs/design/prefix_caching.md`
  Focus: design goals, hash-based caching model, cache lifecycle, and examples.
- `vllm/vllm/v1/engine/core.py`
  Focus: where request block hasher is initialized and attached to new `Request` objects.
- `vllm/vllm/v1/request.py`
  Focus: `Request` fields such as `block_hashes`, `num_computed_tokens`, `num_cached_tokens`, and when hashes are updated.
- `vllm/vllm/v1/core/kv_cache_utils.py`
  Focus: block-hash generation, `extra_keys`, `cache_salt`, and "full blocks only" behavior.
- `vllm/vllm/v1/core/kv_cache_manager.py`
  Focus: local prefix lookup, `allocate_slots()`, `cache_blocks()`, `free()`, and `reset_prefix_cache()`.
- `vllm/vllm/v1/core/kv_cache_coordinator.py`
  Focus: single-group versus hybrid-group cache-hit coordination and alignment logic.
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
  Focus: `find_longest_cache_hit()`, `touch`, block allocation rules, and per-attention-type behavior.
- `vllm/vllm/v1/core/block_pool.py`
  Focus: cached block map, free queue, ref-count changes, eviction, and prefix-cache reset semantics.
- `vllm/vllm/model_executor/models/deepseek_v2.py`
  Focus: `DeepseekV3ForCausalLM`, MLA path selection, and DeepSeek-specific KV-cache spec details.

### `vllm-ascend/`

- `vllm_ascend/core/recompute_scheduler.py`
  Focus: where local and external prefix hits are merged into `num_computed_tokens`.
- `vllm_ascend/core/scheduler_dynamic_batch.py`
  Focus: alternate Ascend scheduler path that also integrates prefix hits.
- `vllm_ascend/worker/model_runner_v1.py`
  Focus: DeepSeek MLA KV-cache tensor layout handling on Ascend workers.
- `vllm_ascend/utils.py`
  Focus: block-size refresh logic and prefix-cache/chunked-prefill related adjustments.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
  Focus: hash-based external prefix lookup using `request.block_hashes`.
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`
  Focus: CPU-side second-level prefix cache and its hit-rate logging.
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`
  Focus: how CPU prefix hits are translated into `num_external_computed_tokens`.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector.py`
  Focus: UCM external prefix-cache adapter entry points.
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
  Focus: remote prefill / remote decode flow and how it differs from pure hash lookup.
- `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`
  Focus: how multiple connectors are coordinated in one deployment.
- `docs/source/tutorials/models/DeepSeek-V3.1.md`
  Focus: deployment examples, recommended flags, and the note about enabling prefix caching by removing `--no-enable-prefix-caching`.
