# Runtime Scheduler Reconfiguration

This document describes a proposed runtime API for changing selected vLLM
scheduler parameters without restarting the serving endpoint. The target use
case is benchmarking and asynchronous RL workflows where the model, parallelism,
and memory layout stay fixed, but each run needs different scheduler admission
limits.

## Summary

The API is intentionally narrower than general engine reconfiguration:

1. Pause generation with `pause_generation(mode="keep")`.
2. Validate that the scheduler is fully paused and has no unfinished requests.
3. Apply a small allowlist of scheduler-only parameters.
4. Resume generation with `resume_generation()`.

Example Python usage:

```python
await engine.pause_generation(mode="keep")
await engine.reconfigure_scheduler(
    max_num_batched_tokens=32768,
    max_num_seqs=128,
    max_num_scheduled_tokens=32768,
)
await engine.resume_generation()
```

Example HTTP usage:

```bash
curl -fsS -X POST "http://localhost:8000/pause?mode=keep"

curl -fsS -X POST "http://localhost:8000/reconfigure" \
  -H "Content-Type: application/json" \
  -d '{"max_num_batched_tokens": 32768, "max_num_seqs": 128}'

curl -fsS -X POST "http://localhost:8000/resume"
```

This API should not be used to change model loading, distributed topology, KV
cache layout, compilation, CUDA graph, or weight-format settings.

## Safety Contract

`reconfigure_scheduler()` must fail closed unless all of the following are true:

- The engine is paused with `mode="keep"`, which maps to `PAUSED_ALL`.
- The scheduler has no unfinished requests.
- All requested parameters are in the runtime-reconfigurable allowlist.
- The new values pass the same consistency checks as startup configuration where
  applicable.

The intended benchmark pattern is to launch vLLM once with the largest required
static capacity, then dynamically lower or tune scheduler admission limits between
benchmark cases. Increasing limits above the capacity assumed at startup is not
safe unless every dependent subsystem has already been initialized for that
ceiling.

## Initial API Surface

The request object supports scheduler admission/token limits and prefill knobs:

```python
@dataclass
class SchedulerReconfigureRequest:
    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_num_scheduled_tokens: int | None = None
    enable_chunked_prefill: bool | None = None
    long_prefill_token_threshold: int | None = None
```

Future extensions can add additional scheduler policy knobs:

```python
    max_num_partial_prefills: int | None = None
    max_long_partial_prefills: int | None = None
    scheduler_delay_factor: float | None = None
    scheduling_policy: Literal["fcfs", "priority"] | None = None
```

## Engine Argument Support Matrix

The following table classifies vLLM engine arguments from the public engine-args
documentation, with emphasis on arguments used by InferenceX benchmark scripts.

| Argument | Runtime support | Rationale |
| --- | --- | --- |
| `--max-num-seqs` | Supported | Scheduler admission limit; update `SchedulerConfig.max_num_seqs` and runtime `max_num_running_reqs`. |
| `--max-num-batched-tokens` | Supported with startup-ceiling caveat | Scheduler token budget; safe when dependent buffers/graphs were initialized for the configured ceiling. |
| `--max-num-scheduled-tokens` | Supported with startup-ceiling caveat | Direct scheduler per-step token budget; must not exceed `max_num_batched_tokens`. |
| `--max-num-partial-prefills` | Candidate | Scheduler policy for chunked prefill; can be changed while empty and fully paused. |
| `--max-long-partial-prefills` | Candidate | Scheduler policy for long-prefill fairness; can be changed while empty and fully paused. |
| `--long-prefill-token-threshold` | Candidate | Scheduler classification threshold; can be changed while empty and fully paused. |
| `--scheduler-delay-factor` | Candidate | Scheduler delay policy; low risk while empty and fully paused. |
| `--scheduling-policy` | Candidate with stricter empty-queue checks | Queue ordering changes; must rebuild or require empty waiting/skipped/running queues. |
| `--enable-chunked-prefill` | Later | Changes scheduling semantics; needs targeted correctness tests. |
| `--disable-chunked-mm-input` | Later | Changes multimodal chunking semantics; needs targeted correctness tests. |
| `--preemption-mode` | Later | Interacts with KV/preemption behavior; needs targeted correctness tests. |
| `--cuda-graph-sizes` | Restart required | Affects CUDA graph capture and compilation assumptions. |
| `--num-lookahead-slots` | Restart required | Affects speculative decoding KV slots and runtime buffers. |
| `--scheduler-cls` | Restart required | Changes scheduler implementation. |
| `--disable-hybrid-kv-cache-manager` | Restart required | Affects KV cache manager allocation strategy. |
| `--async-scheduling` | Restart required | Changes scheduling/execution architecture. |
| `--tensor-parallel-size` | Restart required | Changes distributed topology and model partitioning. |
| `--data-parallel-size` | Restart required | Changes distributed topology and request routing. |
| `--enable-expert-parallel` | Restart required | Changes MoE routing and distributed topology. |
| `--gpu-memory-utilization` | Restart required | Used to size GPU/KV memory allocations. |
| `--max-model-len` | Restart required | Affects model limits, cache sizing, and validation. |
| `--block-size` | Restart required | Affects KV cache block layout. |
| `--kv-cache-dtype` | Restart required | Affects KV cache storage dtype and allocation. |
| `--quantization` | Restart required | Affects model loading and kernel selection. |
| `--served-model-name` | Restart required or serving-layer only | Serving metadata; not scheduler behavior. |
| `--model-loader-extra-config` | Restart required | Affects model loading. |
| `--max-cudagraph-capture-size` | Restart required | Affects CUDA graph capture and compilation assumptions. |
| `--enable-prefix-caching` / `--no-enable-prefix-caching` | Restart required | Affects cache manager behavior and cached state. |
| `--compilation-config` | Restart required | Affects `torch.compile` and CUDA graph configuration. |
| `--speculative-config` | Restart required | Affects speculative decoding workers, buffers, and scheduling assumptions. |
| `--kv-transfer-config` | Restart required | Affects distributed KV transfer setup. |
| `--kv-events-config` | Restart required | Affects event publishing setup. |

## InferenceX Integration

InferenceX benchmark scripts commonly start a server once, wait for readiness,
and call `run_benchmark_serving` from `benchmarks/benchmark_lib.sh`. The runtime
reconfiguration hook should live in that shared wrapper so individual benchmark
scripts can opt in with environment variables.

Suggested environment variables:

```bash
VLLM_DYNAMIC_RECONFIGURE=1
VLLM_MAX_NUM_BATCHED_TOKENS=32768
VLLM_MAX_NUM_SEQS=128
VLLM_MAX_NUM_SCHEDULED_TOKENS=32768
VLLM_MAX_NUM_PARTIAL_PREFILLS=4
VLLM_MAX_LONG_PARTIAL_PREFILLS=1
VLLM_LONG_PREFILL_TOKEN_THRESHOLD=8192
VLLM_SCHEDULER_DELAY_FACTOR=0.0
VLLM_SCHEDULING_POLICY=fcfs
```

Suggested shell helper:

```bash
reconfigure_vllm_scheduler() {
    local port="$1"
    local base_url="http://0.0.0.0:$port"

    # Build JSON body from set env vars
    local json="{"
    local sep=""
    [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]] && \
        json+="${sep}\"max_num_batched_tokens\":${VLLM_MAX_NUM_BATCHED_TOKENS}" && sep=","
    [[ -n "${VLLM_MAX_NUM_SEQS:-}" ]] && \
        json+="${sep}\"max_num_seqs\":${VLLM_MAX_NUM_SEQS}" && sep=","
    json+="}"

    curl -fsS -X POST "$base_url/pause?mode=abort&clear_cache=true"
    curl -fsS -X POST "$base_url/reconfigure" \
        -H "Content-Type: application/json" -d "$json"
    curl -fsS -X POST "$base_url/resume"
}
```

`run_benchmark_serving` can call the helper before launching the benchmark client:

```bash
if [[ "${VLLM_DYNAMIC_RECONFIGURE:-0}" == "1" && "$backend" == "vllm" ]]; then
    reconfigure_vllm_scheduler "$port"
fi
```

A single-server sweep can then vary scheduler settings between runs:

```bash
for conc in 1 2 4 8 16 32 64 128; do
    export VLLM_DYNAMIC_RECONFIGURE=1
    export VLLM_MAX_NUM_SEQS="$conc"
    export VLLM_MAX_NUM_BATCHED_TOKENS=32768
    export VLLM_MAX_NUM_SCHEDULED_TOKENS=32768

    run_benchmark_serving \
      --model "$MODEL" \
      --port "$PORT" \
      --backend vllm \
      --input-len "$ISL" \
      --output-len "$OSL" \
      --random-range-ratio "$RANDOM_RANGE_RATIO" \
      --num-prompts "$((conc * 10))" \
      --max-concurrency "$conc" \
      --result-filename "${RESULT_FILENAME}_conc${conc}" \
      --result-dir /workspace/
done
```

InferenceX result JSON should record the dynamic scheduler values and whether the
server was restarted. This keeps results comparable across benchmark runs.

Example metadata:

```json
{
  "dynamic_reconfigure": true,
  "server_restarted": false,
  "vllm_max_num_batched_tokens": 32768,
  "vllm_max_num_seqs": 128,
  "vllm_max_num_scheduled_tokens": 32768
}
```

## Open Questions

- Should runtime increases above startup values be rejected universally, or only
  for parameters known to affect static buffers and graph capture?
- Should the API return the effective scheduler configuration after applying
  defaults and validation?
- Should `scheduling_policy` be supported by rebuilding empty request queues, or
  should it remain restart-required until there is a stronger use case?
