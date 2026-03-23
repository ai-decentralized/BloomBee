## Summary

This PR improves the stage-to-stage inference path and cleans up several runtime hot-path and shutdown issues.

It combines four changesets:

1. `Optimize downstream decode push path`
2. `Clean inference metadata and runtime cleanup`
3. `Restore speculative pruner startup path`
4. speculative-decoding stability fixes validated locally

The current focus is same-machine validation for a cross-machine-oriented codebase. The changes therefore avoid adding local-only fast paths and instead reduce overhead in the existing BloomBee + Hivemind transport/runtime path.

## What changed

### 1. Reduce downstream decode-path contention

- Keep downstream decode aligned with upstream `rpc_push` once the session is warm.
- Reuse the already-open downstream inference stream instead of repeatedly sending equivalent decode requests on the client path.
- Preserve an opt-in hook for server-to-server output compression experiments.

### 2. Trim redundant `rpc_inference` metadata work

- Stop building and sending `args_structure` for the `rpc_inference` hot path, since inference already uses a fixed positional tensor layout.
- Keep `rpc_forward` / `rpc_backward` packaging unchanged.
- Fix legacy `rpc_inference` metadata call sites that accidentally passed `span_uids` in the `args_structure` position.
- Avoid serializing `args_structure=None` into request metadata.

### 3. Clean up stale runtime code

- Remove dead speculative-training branches in `TransformerBackend`.
- Remove unused tensor-hash debug code.
- Remove unused streaming/micro-batch state from `TransformerConnectionHandler`.
- Remove an unused file-polling helper for micro-batch IPC.
- Remove unused `args_structure` propagation in stage-to-stage micro-batch push metadata.

### 4. Fix runtime diagnostics / shutdown behavior

- Fix a KV verbose-log bug in `KVCacheManager.update_cache()` that referenced a non-existent helper.
- Replace backend shutdown parameter clearing with a dtype/device-safe release path.
- Backend shutdown no longer emits the previous incompatible-tensor-type warnings during local validation.

### 5. Keep current local validation knobs enabled

- Default `DEFAULT_MICRO_BATCH_SIZE` is set to `2` for overlap-path testing.
- Server policy keeps KV cache at `50/50` GPU/CPU for current offload validation.

### 6. Restore speculative decoding to a runnable local state

- Restore the LLaMA-7B speculative pruner startup path and make it lazy-initialized instead of hard-failing server startup.
- Fix speculative pruner construction for `SimpleProbabilityPruner`.
- Remove the obsolete `datasets` dependency from `benchmarks/benchmark_speculative_decoding.py`.
- Restore pruned speculative hidden states before downstream processing when batch/sequence shape no longer matches the original draft tree.
- Preserve full speculative verification outputs instead of trimming them back to the committed token count on the client path.
- Add a guarded fallback for empty logits slices during speculative verification.

## Validation

### Local end-to-end inference

Validated multiple times with repo-source execution via:

```bash
PYTHONPATH=/home/user/BloomBee/src
python -m bloombee.cli.run_dht ...
python -m bloombee.cli.run_server ... --block_indices 0:16 ...
python -m bloombee.cli.run_server ... --block_indices 16:32 ...
python BloomBee/benchmarks/benchmark_inference.py --model huggyllama/llama-7b --torch_dtype float32 --seq_len 4 --batch_size 2
```

Representative successful result after the cleanup pass:

- `throughput=5.52 tokens/sec/sequence`
- `effective_throughput=11.04 tokens/sec`

### Micro-batching

Validated that:

- `micro_batch_size=2` enables the overlap-only path
- `batch_size=4, micro_batch_size=2` actually splits into 2 micro-batches
- current implementation is still **overlap-only**, not KV-memory-saving micro-batching

### KV cache offload

Validated mixed cache placement through runtime logs:

- `Cache GPU%: 50% | CPU%: 50%`
- `cache_device_type=DeviceType.MIXED`
- cache segments split across `CUDA` and `CPU`

### Local speculative decoding

Validated locally with repo-source execution via:

```bash
PYTHONPATH=/home/user/BloomBee/src
python -m bloombee.cli.run_dht ...
python -m bloombee.cli.run_server ... --block_indices 0:16 --batch_size 1 ...
python -m bloombee.cli.run_server ... --block_indices 16:32 --batch_size 1 ...
python BloomBee/benchmarks/benchmark_speculative_decoding.py --model huggyllama/llama-7b --torch_dtype float32 --seq_len 4 --batch_size 1 --n_processes 1
```

Representative successful result after the speculative fixes:

- benchmark completes without the previous server-side position-id crash
- benchmark completes without the previous empty-logits crash in `_extract_best_verified_paths_fixed()`
- `Final result: speed=0.70`
- 4 speculative verification iterations executed successfully in the local test run

## Notes

- This PR does **not** yet replace Hivemind unary protobuf `rpc_push` with a binary streaming path.
- The branch currently includes the active local server policy/testing settings used during validation.
- Speculative decoding is now locally runnable again, but this PR does not yet try to optimize speculative throughput.

## Next steps

1. Continue simplifying speculative request/control state, especially fields still duplicated across tensors and metadata.
2. Revisit why speculative cross-stage transfer is still taking the `rpc_push` path even when the code intends to keep full speculative context.
3. Benchmark speculative decoding with larger `seq_len` / `batch_size` combinations and check whether 50/50 KV offload remains stable.
4. Revisit transport-level optimization for cross-machine runs, especially the unary `rpc_push` path over Hivemind.
