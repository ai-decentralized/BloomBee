## Summary

This PR improves the stage-to-stage inference path and cleans up several runtime hot-path and shutdown issues.

It combines nine changesets:

1. `Optimize downstream decode push path`
2. `Clean inference metadata and runtime cleanup`
3. `Restore speculative pruner startup path`
4. speculative-decoding stability fixes validated locally
5. `Avoid redundant microbatch input cloning`
6. `Use empty optional tensors for decode outputs`
7. `Split decode inference output schema`
8. `Split and slim regular decode input layouts`
9. `Move speculative control flags to metadata`

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

### 7. Remove extra copy / placeholder work in regular decode

- Stop cloning `hidden_states` for every micro-batch slice in the sequential overlap path.
- Return empty dummy tensors for absent decode-only optional outputs instead of scalar placeholder tensors.
- Split regular decode final outputs from speculative final outputs:
  - regular decode now emits a compact 3-tensor prefix
  - speculative decoding keeps the 6-tensor routing prefix
- Update `_push_outputs()` so stage-to-stage forwarding accepts either compact decode outputs or full speculative outputs and reconstructs the downstream request layout correctly.

### 8. Slim regular decode request inputs and harden the speculative pruner

- Split regular decode request inputs away from the old shared speculative-style `rpc_inference` layout.
- Regular decode now uses:
  - a 6-tensor compact layout when prompt/hypothesis payloads are actually present
  - a 4-tensor minimal layout for the common case with no prompt/hypothesis payload
- Server-side full-batch inference now accepts and defaults both compact/minimal regular layouts without changing speculative request handling.
- Fix `SimpleProbabilityPruner` token indexing by casting draft-token ids to integers before indexing logits/probabilities.

### 9. Remove duplicated speculative control tensors from the request path

- Stop sending `need_pruning` and `is_spec_dec` as speculative request tensors.
- Keep those control flags in request metadata instead.
- Add `spec_compact_v1` full-batch parsing on the server side for speculative requests.
- Update stage-to-stage forwarding so speculative `rpc_push` reconstructs the downstream request layout without reintroducing the removed control tensors.
- Preserve `need_pruning_next` semantics by carrying the forwarded value in metadata on the downstream request path.

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

Representative successful result after the later decode-path cleanup/splitting pass:

- `throughput=5.65 tokens/sec/sequence`
- `effective_throughput=22.59 tokens/sec`

Representative successful result after the regular-decode input layout split:

- `throughput=5.64 tokens/sec/sequence`
- `effective_throughput=22.55 tokens/sec`

Representative successful result after the speculative control-flag cleanup:

- short local speculative benchmark still completes successfully
- `Final result: speed=0.67`

Representative successful result after rerunning regular inference with a longer decode window:

- `seq_len=64`, `batch_size=1`
- `throughput=7.66 tokens/sec/sequence`
- full generated text remains coherent in the local end-to-end check

### Micro-batching

Validated that:

- `micro_batch_size=2` enables the overlap-only path
- `batch_size=4, micro_batch_size=2` actually splits into 2 micro-batches
- removing eager micro-batch `hidden_states.clone()` does not break local 2-server inference
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

Representative successful result after the later decode-schema split:

- speculative benchmark still completes successfully with the compact regular-decode output schema in place
- `Final result: speed=0.56`

Representative successful result after the later regular-decode input split and pruner fix:

- speculative benchmark still completes successfully after the compact/minimal regular-decode input layouts
- `Final result: speed=1.12`

Representative successful result after moving speculative control flags to metadata:

- speculative benchmark still completes successfully with `spec_compact_v1`
- `Final result: speed=0.67`

## Notes

- This PR does **not** yet replace Hivemind unary protobuf `rpc_push` with a binary streaming path.
- The branch currently includes the active local server policy/testing settings used during validation.
- Speculative decoding is now locally runnable again, but this PR does not yet try to optimize speculative throughput.
- Regular decode input/output tensors are slimmer, but the regular path still carries a small routing/control prefix for downstream compatibility.
- Speculative requests now rely on metadata for control flags, but speculative output/routing tensors are still more complex than regular decode.

## Next steps

1. Continue simplifying speculative request/control state, especially the remaining routing/output tensors that are still speculative-specific.
2. Revisit why speculative cross-stage transfer is still taking the `rpc_push` path even when the code intends to keep full speculative context.
3. Benchmark speculative decoding with larger `seq_len` / `batch_size` combinations and check whether 50/50 KV offload remains stable.
4. Revisit whether the remaining regular-decode routing prefix can be reduced further without complicating downstream stage forwarding.
5. Revisit transport-level optimization for cross-machine runs, especially the unary `rpc_push` path over Hivemind.
