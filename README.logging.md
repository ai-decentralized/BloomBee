# BloomBee Logging Reference

BloomBee emits a lot of timing/throughput-related log lines from three
different processes (client, server, DHT) and many timer scopes. The same
word — *latency*, *throughput*, *bytes* — can mean very different things
depending on which file emitted the line. This document is the source of
truth for what each log tag measures, where it's emitted, and which ones
are comparable to each other.

If you only remember one thing: **client `[STEP_LATENCY]` and server
`[SERVER_PROCESSING_LATENCY]` are NOT the same number and are not
directly comparable.**

---

## 1. Clock domains

| Domain | Starts at | Ends at | Emitted by |
|---|---|---|---|
| **Client step latency** | client calls `step()` | client receives the next token | `benchmarks/benchmark_inference.py` |
| **Server processing** | `rpc_inference` receives a request | response is returned to caller | `server/handler.py` |
| **Server-to-server push** | sender submits the serialized payload | receiver `rpc_push` acknowledges | `server/handler.py` (S2S\_\* tags) |
| **Per-stage block timer** | enter `_run_block_forward` | exit `_run_block_forward` | `server/backend.py` |
| **Wire compression** | serializer sees the raw tensor | deserializer hands back the raw tensor | `utils/lossless_transport.py` |

Two server-side events are **never directly observed by the client** —
the client only sees the wall-clock latency of its `step()` call. Two
client-side events are **never directly observed by the server** — the
server has no idea when the client started measuring.

---

## 2. Client-side logs

All emitted by `benchmarks/benchmark_inference.py`.

| Tag | What it measures | Unit | Gotcha |
|---|---|---|---|
| `[STEP_LATENCY]` | One `step()` call end-to-end: client send → client receive | ms | Includes network RTT, server compute, serialize/deserialize. Not decomposable from the client side. |
| `[OVERALL_LATENCY]` | Aggregate of `[STEP_LATENCY]` over the whole run: mean / median / p95 / p99 / min / max | ms | Warmup steps already excluded. |
| `[EARLY_TERMINATE]` | Throughput at the moment early-termination triggered | tok/s | Cumulative mean from step 0 up to here, not an instantaneous reading. |
| `Final result: throughput=X tokens/sec/sequence` | Whole-run **per-sequence** throughput | tok/s/seq | Does **not** multiply by `batch_size`. |
| `Final result: ... effective_throughput=Y tokens/sec` | Whole-run **aggregate** throughput | tok/s | Equals `throughput × batch_size`. |
| `[CLIENT_METRIC_NOTE]` | Self-documenting note printed at the end of the run | — | Reminds you not to compare client latency to server-side per-stage tags. |

---

## 3. Server-side per-step logs

These fire **once per inference step (per worker)**. Values are
instantaneous unless the tag itself says "aggregate".

| Tag | What it measures | Where it's measured |
|---|---|---|
| `[NETWORK_RX]` | Size of the incoming request and how long the handler took to fully receive it | inside `rpc_inference`, measured by the **receiver** |
| `[NETWORK_TRANSFER_LATENCY]` | One-direction S1→S2 wire time | receiver minus sender timestamp |
| `[SERVER_PROCESSING_LATENCY]` | Total time spent inside `rpc_inference` (deserialize + forward + serialize) | sender's wall clock; does **not** include network |
| `[COMM_BREAKDOWN]` | The most informative line: per-step decomposition of the communication path | partly sender, partly receiver — see decomposition below |
| `[STEP_PROFILE]`, `[STEP_TIMING_BREAKDOWN]` | Internal per-step breakdown, includes `queue_wait` | server runtime |
| `[S2S_PUSH_BREAKDOWN]` | Sender-side push decomposition | sender only |

### `[COMM_BREAKDOWN]` field meanings

```
[COMM_BREAKDOWN] step_id=… to_blocks=14:28
  T(GPU→CPU)=0.45ms     ← sender: post-compute D2H copy into pinned host buffer
  T(CPU→NIC)=0.18ms     ← sender: serialize + push into the kernel network stack
  T(NIC→NIC)=2.17ms     ← wire-only time: receiver_recv_ts − sender_send_ts minus receiver processing
  push_e2e=4.49ms       ← sender_send_ts → receiver_ack_ts (includes receiver processing)
  recv_proc=2.32ms      ← receiver: deserialize + unpack
  total_comm=2.80ms     ← T(GPU→CPU) + T(CPU→NIC) + T(NIC→NIC); NOT equal to push_e2e
  compute=51.78ms       ← worker's GPU compute time for this step
  critical_path: compute=94.9% comm=5.1%   ← inspect this to identify the bottleneck
  BW(NIC)=4.1MB/s       ← wire_bytes / T(NIC→NIC); effective payload bandwidth
  wire_bytes=8853       ← actual bytes pushed (post-compression)
```

---

## 4. Server-side run-end aggregate logs

Emitted once at the end of the run. Values are means across steps
unless explicitly marked median / p95.

| Tag | Purpose | Comparable to |
|---|---|---|
| `[TIMING_TABLE]` | Human-readable summary row | each other |
| `[PAPER_TIMING_TABLE]` | Same shape as `[TIMING_TABLE]` but on downstream stages it prefers **upstream sender + wire** fields. This is the canonical "paper-table" row. | each other |
| `[TIMING_SUMMARY]` | Multi-line block with mean / median / p95 / max per stage + `summary:` row + `ratio:` row | self |
| `[PIPELINE_COMPONENT_VIEW]` | Raw local segment durations (one number per segment). **Does not account for pipeline overlap.** | self only — do not sum the RAW fields and expect end-to-end |
| `[PIPELINE_EXPOSED_VIEW]` | Per-segment time actually exposed on the critical path (overlap-aware) | **these are the additive ones** |
| `[PIPELINE_GPU2GPU]` | Two GPU↔GPU measurements: `T_GPU->GPU_PIPE` (full network path) vs `T_GPU->GPU_PURE` (same-host cuda↔cuda only) | self |
| `[TIMING_NOTE]` | Self-documenting explanation of the field names | — |

### Run-end fields that are most often confused

| Field | What it is | What it is NOT |
|---|---|---|
| `NetLatency` | mean of `push_e2e_ms` over all steps | not a ping; not `T_NIC->NIC` |
| `T_NIC->NIC` | wire-only time (push_e2e minus receiver processing) | not `push_e2e`; not NIC hardware latency |
| `push_e2e` | sender send → receiver ack (includes receiver processing) | not `T_NIC->NIC`; not single-direction |
| `NetBandwidth=…MB/s` | `payload_bytes × 8 / T_NIC->NIC / 1e6` — effective payload bandwidth | not the physical link bandwidth (excludes IP/TCP headers) |
| `*_RAW` (`[PIPELINE_COMPONENT_VIEW]`) | absolute duration of each segment, ignoring overlap | **do not sum these** — pipeline stages run concurrently, RAW double-counts |
| `*_EXPOSED` (`[PIPELINE_EXPOSED_VIEW]`) | overlap-aware exposed time per segment | safe to sum — equals end-to-end |
| `T_GPU_Compute` | GPU compute time on the sender | does **not** include H2D/D2H |
| `InferenceLatency` | this worker's per-step total time (compute + local IO) | not end-to-end client-observed latency |
| `Throughput` | sender-local tokens / second | will not match the client's `Final result throughput` (the client value also includes the network round-trip) |

---

## 5. Lossless-compression logs

All emitted by `utils/lossless_transport.py`. Both are off by default;
enable with the env vars below.

| Tag | Env var to enable | Scope |
|---|---|---|
| `[COMP_RATIO]` | `BLOOMBEE_COMP_RATIO_PROFILE=1` | per-tensor: `wire_ratio`, `raw_bytes`, `wire_bytes` |
| `[COMP_TIMING]` | `BLOOMBEE_COMP_TIMING_PROFILE=1` | aggregated over the run: serialize / wrap / compress / decompress times in ms, and byte counts |

### Byte counts in `[COMP_TIMING]` (commonly confused)

| Field | Definition |
|---|---|
| `serialize_raw_bytes` | bytes of the raw tensor **before** serialization |
| `serialize_wire_bytes` | bytes actually put on the wire (compressed payload + BBLC header) |
| `compress_input_bytes` | bytes that entered the compressor (≈ raw) |
| `compress_output_bytes` | bytes that left the compressor (= wire − header) |
| `serialize_ratio` | `serialize_wire_bytes / serialize_raw_bytes` — **smaller is better** |
| `serialize_savings` | `1 − serialize_ratio` — **larger is better** |

---

## 6. Quick reference: which log answers which question?

| Question | Use this |
|---|---|
| How fast is the whole system end-to-end? | client `[STEP_LATENCY]` mean / `[OVERALL_LATENCY] Mean` |
| How fast is a single worker's local processing? | server `[SERVER_PROCESSING_LATENCY]` (per-step) or `InferenceLatency` (run-end) |
| How slow is one network hop? | `T_NIC->NIC` (run-end aggregate) or `[COMM_BREAKDOWN] T(NIC→NIC)` (per-step) |
| What's the total cost of one push (incl. receiver processing)? | `push_e2e` (per-step) or `NetLatency` (run-end) |
| What's the actual lossless compression ratio? | `[COMP_TIMING] serialize_ratio` (smaller = better) |
| Can I sum these per-segment fields to get end-to-end? | Only the `*_EXPOSED` fields in `[PIPELINE_EXPOSED_VIEW]`. The `*_RAW` fields in `[PIPELINE_COMPONENT_VIEW]` overlap and will overcount. |

---

## 7. Field names accepted by `parsers/parse_timing.py`

The canonical field names the parser writes to its outputs are:

```
T_GPU_to_CPU_ms, T_CPU_to_NIC_ms, T_NIC_to_NIC_ms,
T_NIC_to_CPU_ms, T_CPU_to_GPU_ms,
InferenceLatency_ms, Throughput_tok_s, CommunicationVolume_KB,
T_GPU_Compute_ms, NetworkLatency_ms,
NetworkBandwidth_MBps, NetworkBandwidth_Mbps
```

The parser's `KEY_MAP` accepts several spellings — `T(GPU→CPU)`,
`T_GPU->CPU`, `gpu_to_cpu`, `t_gpu_cpu` etc. all map to
`T_GPU_to_CPU_ms`. That tolerance is intentional: different emit sites
use different conventions, and the parser normalizes them.

---

## 8. Recap of the most common mistakes

1. **Comparing client `[STEP_LATENCY]` to server `[SERVER_PROCESSING_LATENCY]`.**
   The client number includes the network round-trip; the server number
   does not. They will never match.

2. **Summing the `*_RAW` fields from `[PIPELINE_COMPONENT_VIEW]`** and
   expecting to recover end-to-end latency. Pipeline overlap means
   those numbers double-count. Use the `*_EXPOSED` fields from
   `[PIPELINE_EXPOSED_VIEW]` instead.

3. **Treating `T_NIC->NIC` and `push_e2e` as interchangeable.**
   `push_e2e` includes receiver processing; `T_NIC->NIC` does not.

4. **Quoting `throughput` from the client `Final result` line as the
   per-batch throughput.** It is per-sequence. Multiply by `batch_size`
   (or use `effective_throughput`) for the aggregate number.

5. **Forgetting that compression profiling is OFF by default.** If you
   don't see `[COMP_TIMING]` or `[COMP_RATIO]` in your logs, you need
   to set `BLOOMBEE_COMP_TIMING_PROFILE=1` / `BLOOMBEE_COMP_RATIO_PROFILE=1`
   before starting the worker.
