## Summary

This PR consolidates the current BloomBee runtime cleanup and inference-path work on top of upstream `fdff45c`.

It focuses on three areas:

1. slimming the regular decode request/response path
2. restoring and hardening speculative decoding and micro-batching
3. removing stale compatibility layers from inference and forward/backward RPC paths

This document reflects the **final tree state** on `main`, not intermediate experiments that were later reverted.

## Final change set

### 1. Downstream decode now follows upstream push more directly

- Downstream decode reuses the already-open inference session stream instead of repeatedly sending equivalent decode requests.
- The downstream steady-state path is aligned with upstream `rpc_push`.
- An opt-in hook for server-to-server output compression experiments is preserved.

### 2. Regular decode no longer uses the old speculative-style `rpc_inference` request layout

- Regular decode was split away from the shared speculative request layout.
- The regular path now uses compact layouts instead of the old large shared tensor bundle.
- Regular decode no longer carries speculative-only prompt/hypothesis placeholders in the hot path.

### 3. Control flags were moved out of regular/spec request tensors

- `need_pruning` and `is_spec_dec` are no longer sent as regular/spec request tensors.
- These flags now travel in metadata instead.
- Regular decode pruning state was also removed from the output tensor path and kept metadata-only.

### 4. Decode outputs were split from speculative outputs

- Regular decode final outputs were split away from speculative routing outputs.
- The regular path now uses a smaller decode output schema.
- Forward/inference schema handling is now explicitly separated instead of sharing one drifting superset.

### 5. Legacy inference compatibility layers were removed

- The old legacy `rpc_inference` layout fallback was removed.
- Unused decode `v1` layout fallbacks were removed.
- Old compatibility plumbing around `args_structure` in the inference hot path was removed.

### 6. Forward/backward RPC no longer depends on `args_structure`

- `args_structure` was removed from the forward/backward request metadata path.
- Client forward/backward calls now send fixed flat tensor layouts directly.
- Server-side `rpc_forward` / `rpc_backward` no longer unpack requests via `args_structure`.
- The old packaging helpers used only for this path were removed.

### 7. Forward/backward runtime behavior was hardened after removing that compatibility layer

- `RemoteSequential` now tolerates LLaMA inference-only kwargs in the autograd path instead of asserting.
- `PipelineParallelWrapper` is callable, so it behaves correctly under `ModuleBackend`.
- Forward/backward RPC was normalized back to the intended single-output hidden-state contract for training/forward benchmarks.

### 8. Speculative decoding was restored to a stable local state

- The LLaMA speculative pruner startup path was restored and made lazy-initialized.
- Session recovery issues in speculative decoding were fixed.
- Pruned speculative hidden states are restored before downstream processing when needed.
- Verification no longer trims speculative outputs incorrectly on the client path.
- Empty-logit edge cases in verification were guarded.
- The speculative pruner indexing path was fixed for integer token ids.

### 9. Micro-batch and speculative micro-batch execution were hardened

- Redundant `hidden_states.clone()` work in the micro-batch slice path was removed.
- Micro-batch merge handling was fixed for padded `keep_indices` and 3D hidden-state restoration.
- Speculative + micro-batching now runs successfully again in local end-to-end validation.
- A regular micro-batch push bug was fixed so micro-batches are actually forwarded instead of stalling before push.

### 10. Request / push metadata was trimmed without removing paper-useful timing fields

- Default request metadata fields are now omitted when they carry default values.
- Forwarded full-batch push metadata no longer carries stale local `_...` fields that are recomputed at each hop.
- Timing fields used for paper measurements were intentionally preserved.

### 11. Runtime cleanup / shutdown issues were fixed

- A KV verbose-log bug in cache update handling was fixed.
- Backend shutdown now releases parameters/buffers safely without the previous incompatible tensor warnings.
- Several dead helpers and stale runtime branches were removed.

### 12. Historical dead code was removed

- Removed unused `inference_session_time_range.py`
- Removed unused micro-batch polling helpers
- Removed unused streaming/request context state
- Removed unused hash/debug helpers

## Validation

All validation below was run from source with:

```bash
PYTHONPATH=/home/user/BloomBee/src
```

and a local 2-server setup:

```bash
python -m bloombee.cli.run_dht ...
python -m bloombee.cli.run_server ... --block_indices 0:16 ...
python -m bloombee.cli.run_server ... --block_indices 16:32 ...
```

### Forward / backward

The forward/backward path was revalidated after removing `args_structure`:

- `benchmark_forward.py`: `Final result: speed=3.97`
- `benchmark_training.py`: `Final result: fwd_speed=3.92 bwd_speed=6.67`

### Regular decode

Representative local regular-decode result on the current tree:

- default config: no micro-batching, no KV cache offload
- `seq_len=64`, `batch_size=1`
- `Final result: throughput=5.84 tokens/sec/sequence, effective_throughput=5.84 tokens/sec`

Representative local regular micro-batch result from the explicit validation path:

- `seq_len=16`, `batch_size=4`
- `Final result: throughput=5.24 tokens/sec/sequence, effective_throughput=20.97 tokens/sec`

### Speculative decoding

Representative local speculative micro-batch result on the current tree:

- `batch_size=4`, `seq_len=128`
- `Final result: speed=3.75`

### Error scan

The latest local regression pass completed without new:

- `P2PHandlerError`
- `TimeoutError`
- `Dimension mismatch`
- `Failed to push micro-batch`
- Python traceback / assertion failures

## Current runtime defaults and assumptions

- micro-batching is **disabled by default**
- server policy currently keeps:
  - weights on GPU
  - activations on GPU
  - KV cache on GPU (`100% GPU / 0% CPU`)
- micro-batching and speculative micro-batching were validated explicitly with opt-in settings even though the default is now off

## What is still not done

- Hivemind unary `rpc_push` is still used; this PR does **not** replace it with a binary streaming path.
- Cross-machine transport is still paying the BloomBee serialization + Hivemind protobuf/daemon overhead.
- The remaining major optimization work is still transport-level or compute-level, not request-layout cleanup.

## Suggested next steps

1. Revisit transport-level optimization for cross-machine runs, especially unary `rpc_push` over Hivemind.
2. Evaluate client-side `lm_head` / sampling placement if single-stream decode throughput is still limited by client-side work.
3. Benchmark speculative decoding with larger `seq_len` / `batch_size` combinations under the now-stable micro-batch path.
4. Re-evaluate quantization / kernel-side compute optimization, since the regular local decode path is now more compute-bound than metadata-bound.
