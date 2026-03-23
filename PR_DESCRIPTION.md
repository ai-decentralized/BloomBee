## Summary

This PR improves the stage-to-stage inference path and cleans up several runtime hot-path and shutdown issues.

It combines two changesets:

1. `Optimize downstream decode push path`
2. `Clean inference metadata and runtime cleanup`

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

## Notes

- This PR does **not** yet optimize speculative decoding specifically.
- This PR does **not** yet replace Hivemind unary protobuf `rpc_push` with a binary streaming path.
- The branch currently includes the active local server policy/testing settings used during validation.

## Next steps

1. Restore and verify the LLaMA-7B speculative pruner path if it was commented out.
2. Run `benchmarks/benchmark_speculative_decoding.py` after each spec-decoding change.
3. Continue removing duplicated control-plane state in inference requests, especially fields that are currently sent as both tensors and metadata.
4. Revisit transport-level optimization for cross-machine runs, especially the unary `rpc_push` path over Hivemind.
