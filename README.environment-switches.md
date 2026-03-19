# BloomBee Environment Switches

BloomBee has a lot of runtime switches behind `BLOOMBEE_*` environment variables. This file puts them in one place.

- Export them before starting the relevant process (`run_dht`, `run_server`, client, or benchmarks).
- Unless noted otherwise, boolean flags use `1` / `0`.
- Debug helpers also accept `true/false/yes/no/on/off`.

## Core Runtime

| Variable | Default | What it does |
|---|---|---|
| `BLOOMBEE_CACHE` | `~/.cache/bloombee` | Cache directory for downloaded blocks and local BloomBee artifacts. |
| `BLOOMBEE_LOGGING` | `True` | Set to `0` / `false` to disable BloomBee's logging initialization tweaks. |
| `BLOOMBEE_ASYNCIO_LOGLEVEL` | `FATAL` (or `DEBUG` if Hivemind is already in debug mode) | Overrides the asyncio logger level. |
| `BLOOMBEE_IGNORE_DEPENDENCY_VERSION` | unset | Skips the `transformers>=4.43.1,<4.44.0` version assertion. |
| `BLOOMBEE_MAX_RETRIES` | unset | Client-side retry limit before raising an exception. |

## Micro-Batching, Overlap, and Server-to-Server Push

| Variable | Default | What it does |
|---|---|---|
| `BLOOMBEE_ENABLE_MICROBATCH_PIPELINE` | `1` | Master switch for micro-batching. |
| `BLOOMBEE_MICRO_BATCH_SIZE` | `2` | Micro-batch size used for overlap scheduling. |
| `BLOOMBEE_MICRO_ENABLE_GPU_MULTIPLEXING` | parsed, but currently effectively `0` | Intended to shrink active GPU working slots. Right now BloomBee forces `overlap_only`, so this is effectively a no-op. |
| `BLOOMBEE_ENABLE_CROSS_STAGE_PUSH` | `1` | Enables actual server-to-server micro-batch push. Set `0` for dry-run / fallback behavior. |
| `BLOOMBEE_ENABLE_IMMEDIATE_MB_QUEUE` | `1` | Enables immediate in-memory micro-batch queuing on the downstream server. |
| `BLOOMBEE_STORE_MB_FILES_IN_IMMEDIATE` | `0` | In immediate mode, also persist micro-batches to files for legacy compatibility. |
| `BLOOMBEE_MB0_SEMAPHORE_BYPASS` | `1` | Lets MB0 bypass the push limiter to reduce pipeline startup bubble. |
| `BLOOMBEE_MBPIPE_VERBOSE` | `0` | Emits full per-micro-batch logs on every step. |
| `BLOOMBEE_MBPIPE_LOG_EVERY_STEPS` | `16` | Sampling interval for detailed micro-batch logs when verbose mode is off. |
| `BLOOMBEE_MBPIPE_SCHEMA_WARN` | `0` | Escalates micro-batch schema warnings to WARN level. |
| `BLOOMBEE_CLOCK_SYNC_ALPHA` | `0.2` | EMA smoothing factor for cross-server clock sync estimates. |
| `BLOOMBEE_CLOCK_SYNC_MAX_RTT_US` | `2000000` | Maximum RTT, in microseconds, used for clock sync correction. |
| `BLOOMBEE_CLOCK_SYNC_LOG_EVERY` | `64` | Clock sync logging interval. |
| `BLOOMBEE_S2S_STATS_WINDOW` | `32` | Rolling window size for server-to-server telemetry. |
| `BLOOMBEE_S2S_STATS_LOG_EVERY` | `8` | How often S2S telemetry summaries are logged. |

## KV Cache and Offload

| Variable | Default | What it does |
|---|---|---|
| `BLOOMBEE_ENABLE_ASYNC_KV_TRANSFER` | auto | Overrides KV GPU<->CPU transfer mode. If unset, BloomBee enables async KV transfer when micro-batching is enabled. |
| `BLOOMBEE_ENABLE_KV_WAIT_TIMING` | `1` | Enables KV wait timing counters and logs. |
| `BLOOMBEE_VERBOSE_KV_LOGS` | `0` | Restores verbose KV allocation / offload / prefetch logs. Also turns on automatically if the KV debug group is enabled. |

## Lossless Transport and Compression Profiling

| Variable | Default | What it does |
|---|---|---|
| `BLOOMBEE_LOSSLESS_WRAPPER` | `0` | Enables the lossless transport wrapper around serialized tensors. |
| `BLOOMBEE_LOSSLESS_ALGO` | `zstd` | Lossless wrapper algorithm: `zstd`, `zlib`, `zipnn`, or `none`. |
| `BLOOMBEE_LOSSLESS_LEVEL` | `3` | Compression level for the selected lossless wrapper. |
| `BLOOMBEE_LOSSLESS_LAYOUT` | `byte_split` | Wrapper layout: `plain` or `byte_split`. |
| `BLOOMBEE_LOSSLESS_LAYOUT_TARGETS` | `*:*:hidden_states` | `source:channel:tensor_name` selectors for `byte_split` layout. |
| `BLOOMBEE_LOSSLESS_MIN_BYTES` | `49152` | Minimum serialized tensor size before the wrapper even tries to compress. |
| `BLOOMBEE_LOSSLESS_MIN_GAIN_BYTES` | `2048` | Minimum required byte savings, otherwise BloomBee keeps the original buffer. |
| `BLOOMBEE_COMP_RATIO_PROFILE` | `0` | Per-tensor wire ratio logs. Also turns on if the compression debug group is enabled. |
| `BLOOMBEE_COMP_TIMING_PROFILE` | `0` | Compression / decompression timing profile. Also turns on if the compression debug group is enabled. |
| `BLOOMBEE_COMP_DETAIL_PROFILE` | `0` | Heavyweight per-tensor compression diagnostics. Also turns on if the compression debug group is enabled. |
| `BLOOMBEE_COMP_RESEARCH_PROFILE` | `0` | Rolling research-oriented compression summaries. Also turns on if the compression debug group is enabled. |
| `BLOOMBEE_COMP_BIT_PROFILE` | `0` | Bit-level floating-point profiling. Also turns on if the compression debug group is enabled. |
| `BLOOMBEE_COMP_STRIDE_PROFILE` | `0` | Strided / chunk repeat diagnostics. Also turns on if the compression debug group is enabled. |
| `BLOOMBEE_ACT_DIST_PROFILE` | `0` | Activation magnitude histogram profile. Also turns on if the compression debug group is enabled. |
| `BLOOMBEE_COMP_ZIPNN_PROFILE` | `0` | Side-by-side ZipNN diagnostics without changing the actual wire format. |
| `BLOOMBEE_COMP_SUMMARY_LOG_EVERY` | `128` | Log interval for rolling compression summaries. |
| `BLOOMBEE_DEBUG_TENSOR_NAMES` | `hidden_states` | Comma-separated tensor names used by compression debug targeting. `*` matches all. |
| `BLOOMBEE_WIRE_TRUNCATE_FP16` | `0` | Truncates selected FP32 tensors to FP16 before serialization. |
| `BLOOMBEE_WIRE_TRUNCATE_TARGETS` | `client:rpc_inference:hidden_states` | `source:channel:tensor_name` selectors for FP16 wire truncation. |
| `BLOOMBEE_WIRE_TRUNCATE_PHASES` | `prefill,decode,spec_decode` | Which phases allow FP16 wire truncation. |

## Activation Dumping

| Variable | Default | What it does |
|---|---|---|
| `BLOOMBEE_DUMP_ACTIVATIONS` | `0` | Enables server-side dumping of real intermediate activations. |
| `BLOOMBEE_ACTIVATION_DIR` | `/tmp/real_activations` | Output directory for activation dumps. |
| `BLOOMBEE_ACTIVATION_SAMPLES` | `20` | Maximum number of captured activation samples. |
| `BLOOMBEE_ACTIVATION_PHASES` | `prefill,decode` | Comma-separated phases to capture. |

## Debug Groups and Log Channels

`BLOOMBEE_DEBUG=1` is the broad "turn on BloomBee debug mode" switch. The group-level toggles below let you enable only one subsystem. Explicit per-channel flags still win over the group defaults.

| Variable | Default | What it does |
|---|---|---|
| `BLOOMBEE_DEBUG` | `0` | Global BloomBee debug switch. Enables BloomBee `logger.debug(...)` lines and acts as the parent switch for all debug groups. |
| `BLOOMBEE_DEBUG_COMPRESSION`, `BLOOMBEE_DEBUG_COMP` | `0` | Compression debug group. |
| `BLOOMBEE_DEBUG_KV_CACHE`, `BLOOMBEE_DEBUG_KV` | `0` | KV cache / offload debug group. |
| `BLOOMBEE_DEBUG_MICROBATCH`, `BLOOMBEE_DEBUG_MB` | `0` | Micro-batch / overlap debug group. |
| `BLOOMBEE_DEBUG_INFERENCE`, `BLOOMBEE_DEBUG_INF` | `0` | Client inference debug group. |
| `BLOOMBEE_MBPIPE_LOGS` | follows the micro-batch debug group | Controls `[MBPIPE]` logs. |
| `BLOOMBEE_HANDLER_STEP_TIMING_LOGS` | follows the micro-batch debug group | Controls `[HANDLER_STEP_TIMING]` logs. |
| `BLOOMBEE_COMP_ZIPNN_LOGS` | follows the compression debug group | Controls `[COMP_ZIPNN]` logs. |
| `BLOOMBEE_CLIENT_INFERENCE_LOGS` | follows the inference debug group | Controls client inference transport logs such as `[NETWORK_TX]` and `[CLIENT_INFERENCE_END]`. |
| `BLOOMBEE_CROSS_GPU_TRANSFER_LOGS` | follows the micro-batch debug group | Controls cross-GPU transfer logs. |
| `BLOOMBEE_KV_SOURCE_PROBE_LOGS` | follows the KV debug group | Controls `[KV_SOURCE_PROBE]` logs. |

## Refreshing This List

If you add a new switch later, this command is the quickest way to rescan the repository:

```bash
rg -n -o "BLOOMBEE_[A-Z0-9_]+" README.md src benchmarks tests | sort -u
```
