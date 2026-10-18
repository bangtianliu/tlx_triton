# MI450 attention backward: profiling and optimized PR #2709 port

Measured through 2026-10-18 on **MI450 / gfx1250**. The original
[PR #2709](https://github.com/facebookexperimental/triton/pull/2709) reports
**MI350 / gfx950** results; those cross-device numbers remain historical context.

## Current result

The default packed-BF16-dQ policy reaches **1.084–1.161× AITER throughput**
on all ten matched MI450 shapes (17 October table; re-measured on 18 October
within 0.069%). FP32 dQ, which matches AITER's FP32 dQ accumulation, now
reaches **1.017–1.071× AITER throughput** on the same shapes. It uses one base
plus immediate atomic addresses and spreads FP32 atomics through the dK bridge.

All 20 shape/precision pairs pass the 1.2% NRMSE guard, and 125 backward tests
pass. See the [18 October FP32 comparison](#final-fp32-and-packed-comparison)
and [raw results](../../../benchmarks/results/mi450_bwd_fp32dq_20261018.jsonl).
The earlier sections preserve the 16/17 October port and profiling history.

## Historical full-port result (16 October)

The first MI450 implementation retained the phase delay without faithfully
porting the original fragment schedule. This revision implements the remaining mechanisms:
eight dK fragment groups, two independent dQ chains, explicit WMMA ordering,
operand prefetch/residency policies, asynchronous statistics and larger Q/dO
staging tiles, rematerialized coordinates, and packed BF16 dQ atomics with a
WMMA-specific accumulator shuffle. The final drain also uses the two dQ chains.

On the five original PR shapes, complete backward calls are **1.23–1.31×
faster than the first MI450 port**. Latency is now **0.7–7.9% above the published
MI350 results**, versus 30–35% before. AITER remains faster on this MI450.
These cross-device comparisons do not establish the hardware's relative limits.

The new default accumulates dQ in **BF16**, matching the original PR's packed
atomic design. The first port and the gfx1250 AITER baseline accumulate dQ in
FP32. The performance gain therefore includes this precision change. A selectable
FP32 path retains the improved fragment schedule. All outputs are BF16; dK/dV
remain FP32 register accumulators in both TLX policies.

## Original PR shapes: published MI350 versus measured MI450

Hq=64, Hkv=8, D=128, BF16, non-causal. Positive final-column values mean
this MI450 implementation still takes longer than the published MI350 PR.

| B | N | PR, MI350 ms | First port, MI450 ms | Full port, MI450 ms | Gain over first port | Latency vs MI350 |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 1024 | 1.967 | 2.649 | 2.018 | 1.312× | +2.6% |
| 16 | 2048 | 7.174 | 9.551 | 7.737 | 1.234× | +7.9% |
| 16 | 4096 | 27.725 | 36.096 | 28.154 | 1.282× | +1.5% |
| 16 | 8192 | 109.201 | 142.761 | 111.096 | 1.285× | +1.7% |
| 15 | 16384 | 404.942 | 524.615 | 407.933 | 1.286× | +0.7% |

### Matched AITER comparison on MI450

Both providers receive identical inputs and forward state. `TLX/AITER` is a
throughput ratio; below 1 means AITER is faster. The FP32 column changes only
the TLX dQ accumulation policy; it keeps the new scheduling mechanisms.

| N | TLX packed ms | TLX TFLOP/s | TLX FP32 ms | AITER ms | AITER TFLOP/s | TLX/AITER | dQ NRMSE vs AITER |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2.018 | 681.0 | 2.263 | 1.973 | 696.5 | 0.978× | 0.353% |
| 2048 | 7.737 | 710.5 | 8.613 | 6.979 | 787.7 | 0.902× | 0.437% |
| 4096 | 28.154 | 781.1 | 31.982 | 26.438 | 831.8 | 0.939× | 0.556% |
| 8192 | 111.096 | 791.8 | 122.490 | 103.449 | 850.3 | 0.931× | 0.734% |
| 16384 | 407.933 | 808.6 | 450.480 | 372.094 | 886.5 | 0.912× | 0.993% |

The B=15, N=16384 AITER baseline uses B=8 plus B=7 calls. Its unsplit binary
fails correctness beyond its large-workspace addressing boundary; TLX uses
one full-batch call. Both AITER calls and the final GQA reductions are timed.

### B=8 series from the AITER walkthrough

Hq/Hkv=64/8, D=128. Every AITER point here is one batch call.

| N | First port ms | Full port ms | TLX TFLOP/s | AITER ms | TLX/AITER |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1.561 | 1.251 | 549.4 | 1.003 | 0.801× |
| 2048 | 4.760 | 3.842 | 715.5 | 3.504 | 0.912× |
| 4096 | 18.894 | 15.390 | 714.4 | 13.260 | 0.862× |
| 8192 | 72.268 | 55.313 | 795.1 | 52.166 | 0.943× |
| 16384 | 286.084 | 220.505 | 797.8 | 198.156 | 0.899× |

## What was ported

| PR idea | gfx1250 implementation | Selected policy |
|---|---|---|
| Own a KV tile and process its mapped GQA heads | One workgroup per KV tile/head/batch; dK/dV remain in FP32 registers across the Q/head loop | Enabled |
| Transposed scores and dP | K·Qᵀ and V·dOᵀ; Pᵀ/dSᵀ feed gradient products in register ownership | Enabled |
| Delay dQ by one Q tile | Two alternating dS LDS slots, a zero-seeded prologue, and a separate final drain | Enabled |
| Fragment current dK | Eight `(BN/2,32)` accumulator groups, obtained with register-only `tlx.extract_slice` | Enabled |
| Split previous dQ into two chains | Independent `(BM,64)` head-column halves; K=32 reduction bands; the drain uses the same split | Enabled |
| Expose native instruction ordering | WMMA updates separated by matrix-only `amd_sched_barrier(0xff6)` compiler markers | Enabled |
| Prefetch delayed operands during current work | Start previous-dS reads before dK; issue the next dS band and current K operands ahead of their consumers | Enabled |
| Keep reused operands resident | Explicit V and partial-K residency policies, including the original leading 224/256 K rows | Available; slower on this build/device |
| Stage Q/dO and statistics asynchronously | Two-slot TDM rings for Q, dO, LSE and D, with correct final draining | Enabled |
| Load a larger Q/dO outer tile for several compute phases | `outer_m` selects a 64/128-row load tile independently of 16/32/64-row compute phases | Available; selected load/compute tile is 64/64 |
| Rematerialize address coordinates near use | `tlx.rematerialized_range` at dQ publication | Enabled |
| Packed BF16 atomics and final layout conversion | Aligned BF16 pairs in a new native-WMMA accumulator permutation; device-scope buffer atomics; conversion gathers that permutation | Enabled; FP32 option retained |
| Preserve completion and operand liveness | Native WMMA intrinsics retain LLVM hazard tracking; SSA carries live operands; scheduling markers order the independent groups | Adapted from the CDNA MFMA/AGPR contract |

The original CDNA-only `amd_scheduled_mfma` / `amd_mfma_commit` operations reject
gfx1250. This port composes existing WMMA layouts, native dot lowering, register
slices and scheduling markers. It does not reuse gfx950 AGPR assignments or
opaque assembly WMMA instructions. No native compiler changes were needed.

The selected 256-row KV configuration has eight dK groups. Each group lowers
to four native WMMAs at BM=64; each corresponding dQ half-update uses two.
The assembly therefore alternates **4 dK, 2 dQ-half-0, 2 dQ-half-1** instructions
for each of the eight bands. This is group-level ordering adapted to WMMA's
native tile, rather than a claim of one-for-one gfx950 instruction identity.

The ring invariant is unchanged: current dS is published to one slot while
previous dQ reads the other; the workgroup synchronizes before slot reuse.
Matrix scheduling markers add no hardware memory barrier. The WMMA lowering
still emits the required arithmetic hazard waits.

## Packed-atomic layout correction and accuracy

The initial packed attempt faulted with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`. An isolated BF16 buffer atomic
succeeded. The fault was the port's address layout: adjacent registers in this
WMMA dQ layout hold different rows. The old FP32 shuffle did not give those
register pairs adjacent, aligned BF16 addresses.

The corrected permutation places row bit 0 in accumulator bit 0, followed by
the five lane bits in bits 1–5. A wave then issues 32 aligned pairs spanning
128 contiguous bytes. Remaining bits are permuted bijectively, and conversion
loads each logical output's corresponding shuffled address. The final kernel
has `buffer_atomic_pk_add_bf16` instructions; the FP32 variant uses
`buffer_atomic_add_f32`.

Full-shape comparisons against AITER passed the driver's **1.2% NRMSE guard**
for dQ/dK/dV on all ten final shapes. Packed dQ NRMSE grows from about **0.35% at
N=1024 to 0.99% at N=16384**. This is an accumulation-precision tradeoff, not
unchanged accuracy. At the largest shape the FP32 TLX dQ result differs from
AITER by about **0.0035% NRMSE**. Cross-provider dK/dV NRMSE is about 0.26%,
including the providers' different GQA reduction/rounding paths.

FP32-autograd comparisons also covered small MHA, multi-batch GQA, a
three-query-head group, N=384 fallback tiles, repeated workspace reuse, and
the optional resident/outer-tile policies. Accepted tuning runs carry their
reference errors in the raw data. The earlier pytest run predates this
extension; no new pytest run is claimed for the full port.

## Scheduling and residency experiments

At B=8, Hq/Hkv=64/8, N=2048, representative measurements were:

| Change | Complete call ms |
|---|---:|
| First port: delayed, TDM, FP32 dQ | 4.761 |
| Eight groups + two chains + ordered WMMA + rematerialization, FP32 | 4.993 |
| Add asynchronous LSE/D staging | 4.399 |
| Add per-kernel register-allocation controls | 4.275 |
| Add aligned packed BF16 dQ | 3.835 |
| Final two-chain drain / prefetch revision, final remeasurement | 3.842 |
| Final schedule with FP32 dQ, final remeasurement | 4.280 |

These are staged measurements across recorded source revisions, rather than a
factorial attribution of every gain. In the final source, disabling the dQ
delay takes **4.106 ms**, versus **3.842 ms** delayed: **1.069×**.

Explicit VGPR input/output anchors also enforced ordering, but caused more
spilling than the selected matrix-only markers. Larger outer tiles, extra
TDM stages, unrolling, wave scheduling bit 2, and 16-wave configurations were
evaluated. None displaced the selected 64×256 / eight-wave / two-stage policy.
The selected 128-row KV fallback uses a WMMA warp arrangement compatible with
its half-row dK fragments.

Residency modes in the final source, with all other selected features retained:

| Mode | Complete call ms | VGPRs | Private bytes/thread |
|---:|---:|---:|---:|
| 0, stream K/V from LDS | 3.842 | 510 | 12 |
| 1 | 4.676 | 512 | 748 |
| 2 | 5.517 | 512 | 792 |
| 3 | 5.134 | 512 | 532 |
| 4 | 4.065 | 510 | 12 |

Mode 1 keeps score K/V; mode 2 also keeps 3/4 of dQ's K; mode 3 keeps V and the
leading 224 of 256 dQ K rows; mode 4 keeps V alone. The native WMMA build uses
VGPRs for these banks. The extra residency increases spill traffic enough to
lose its LDS savings, so mode 0 remains selected. This is measured policy
selection; the resident implementations remain available.

Selected-kernel code inspection at B=8, Hq/Hkv=64/8, N=2048:

- 8 wave32 waves; Q=64, KV=256; 2 staging slots; no outer-loop unroll.
- 277,440 bytes LDS; 510 VGPRs; 68 SGPRs.
- **12 bytes private storage per thread**; this revision is not spill-free.
- 160 WMMAs in the steady phase plus 32 in the drain, 192 statically.
- 10 static TDM load sites: K/V, four initial staged inputs, four refills.
- 32 static packed atomic instructions, versus 64 FP32 atomics in that variant.

These are code-generation facts, not a profiler-derived attribution of the
remaining AITER or MI350 performance gap.

## API and supported inputs

[Kernel and wrapper](../ops/kernels/flash_attn/gfx1250_bwd.py),
[WMMA layout](../language/tlx/types.py),
[benchmark driver](../../../benchmarks/mi450_fa_bwd.py), and
[AITER C ABI bridge](../../../benchmarks/mi450_bwd_aiter_bridge.cpp).

Inputs are contiguous BHND BF16 Q/K/V/O/dO, FP32 natural-log LSE, D=128,
equal Q/KV sequence lengths, non-causal, positive batch/head counts and
Hq divisible by Hkv. N must be at least 256 and tile-aligned. Automatic
selection covers multiples of 128; the PR's multiples-of-256 contract is
covered. Each query-head group's workspace retains the conservative signed
2-GiB FP32 buffer-offset bound. Tile/stage combinations must fit hardware LDS.

```python
from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import (
    select_config, allocate_workspace, fa_backward,
)

b, hq, n, _ = q.shape
config = select_config(n, b, hq, k.shape[1])    # measured policy, packed BF16 dQ
# config = select_config(n, b, hq, k.shape[1], packed_dq=False)  # FP32 dQ schedule
workspace = allocate_workspace(q, k, config=config)
dq, dk, dv = fa_backward(q, k, v, out, grad_out, lse,
                        config=config, workspace=workspace)
```

`BackwardConfig()` retains the original control settings; `select_config`
provides the measured full-port policy. `allocate_workspace` follows the
selected policy by default, or accepts an explicit config/accumulator dtype.
The benchmark's partial `--config` JSON overrides the selected policy.

## Machine and timing protocol

- MI450 engineering sample, gfx1250, 192 CUs, PCI `0001:04:00.0`, 288 GiB visible HBM. Clocks were not fixed; earlier samples were near 2.35 GHz.
- Checkout base `7eabd13d5`; uncommitted Python changes in this workspace.
- Kernel file SHA256: `bf25907288725f4c9bac798e692c16936d10c5b6ac98d2626c7fea8708ca752c`.
- LLVM revision `a6533ab0f5a1df16ff3dbf7358fb79097c60855e`.
- Container `bangtliu-mi450-bwd`, PyTorch `2.13.0+rocm10.0.0`, ROCm SDK 10.0, editable Triton 3.9.0.
- AITER `c118fafad9ad2873bb63b9b122d59aef5c307a4f`, default gfx1250 ASM; `_perf.co` was not used.
- AITER main binary SHA256 `c86acfc7f1715378b02349a756c20c0fa94ada15f6cb6aa24cfcc1e86ab59cf3`.

Every GPU run used the host `~/.local/bin/gpu-lock` wrapper, which holds
`/tmp/gpu-lock/gpu-0.lock` via Linux `flock`. It was created because no installed
gpu-lock command/service was found. It coordinates cooperating jobs; it is
not a hardware reservation. HIP/CUDA device selection is passed explicitly
through Docker. Process checks showed our workload and the system node agent.

Forward-state generation, allocations and JIT compilation are outside timing.
Both providers use the same inputs/state. Complete calls include fresh dQ
zeroing every time, the main kernel and conversion. AITER additionally includes
its two BF16 GQA output reductions; TLX reduces GQA heads in FP32 registers.

Each result is the median of three `triton.testing.do_bench_cudagraph` trials,
with a 100-ms target graph, at least one call, and ten timed replays per trial.
Raw samples are retained. Useful backward work is counted as
`10*B*Hq*N*N*128 / seconds`. The older walkthrough excluded repeated zeroing;
its historical timings are not the matched baseline here.

### AITER large-workspace correction

The raw unsplit B=15, N=16384 AITER call produced dQ NRMSE about 0.966 against
the first FP32 TLX port. Per-batch checks showed corruption in early batches
and zero dQ in batches 8–14; the FP32 workspace is 7.5 GiB. This is consistent
with a 4-GiB address wrap, but the exact binary instruction was not traced.
Independent small-batch calls matched. The bridge wrapper rebases B=8 and B=7
chunks before running its final GQA reductions. Incorrect unsplit timings
were never accepted as performance results. See the historical experiment logs
and [AITER binary walkthrough](/home/bangtliu/aiter/gfx1250_fmha_bwd_explained.md).

## Raw data and reproduction

- [Final 27 timing records](../../../benchmarks/results/mi450_bwd_full_port_20261016.jsonl): ten packed TLX/AITER pairs, six FP32 TLX points, and one immediate-dQ ablation.
- [Full-port experiments and codegen](../../../benchmarks/results/mi450_bwd_full_port_experiments_20261016.jsonl): configurations, reference errors, source hashes, failed attempts, residency ablations and metadata.
- [First-port timing records](../../../benchmarks/results/mi450_bwd_20261016.jsonl) and [earlier experiments](../../../benchmarks/results/mi450_bwd_experiments_20261016.jsonl) remain historical evidence.

From `/home/bangtliu/tlx_triton`, rebuild the thin AITER bridge if needed:

```bash
docker exec bangtliu-mi450-bwd hipcc -std=c++17 -O2 -shared -fPIC \
  --offload-arch=gfx1250 -DENABLE_CK=0 \
  -I/home/bangtliu/aiter/csrc/include benchmarks/mi450_bwd_aiter_bridge.cpp \
  -L/home/bangtliu/aiter/op_tests/cpp/mha \
  -Wl,-rpath,/home/bangtliu/aiter/op_tests/cpp/mha -lmha_bwd \
  -o /host-tmp/mi450-aiter-bridge.so
```

Run the selected policy under the lock:

```bash
~/.local/bin/gpu-lock run --gpu 0 -- docker exec \
  -e HIP_VISIBLE_DEVICES=0 -e CUDA_VISIBLE_DEVICES=0 \
  -e GPU_LOCK_HELD=/tmp/gpu-lock/gpu-0.lock \
  -e AITER_ASM_DIR=/home/bangtliu/aiter/hsa \
  -e LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib \
  bangtliu-mi450-bwd python benchmarks/mi450_fa_bwd.py \
  --mode bench --shape 8,64,8,2048 --repeat-ms 100 \
  --aiter-library /host-tmp/mi450-aiter-bridge.so \
  --output /host-tmp/mi450-recheck.jsonl
```

Use `--config '{"packed_dq":false}'` for the FP32 policy,
`--config '{"delayed":false}'` for the matched immediate-dQ ablation, or
`--config '{"resident_kv":3}'` for the original-style resident operands.
`--mode check` computes FP32 autograd references; use a small batch/head count.
`--mode compile --artifacts /host-tmp/mi450-asm` saves explicit-target code
without GPU execution. `--mode tune` and `--mode pipeline` preserve the selected
features while changing tile/wave/load or stage/unroll settings.


## 2026-10-17: optimization plan, Phase 0

The preceding tables are the historical 16 October baseline. Work on the
17 October plan is recorded separately in
[Phase 0 evidence](../../../benchmarks/results/mi450_bwd_phase0_20261017.jsonl)
and [Phase 1 measurements](../../../benchmarks/results/mi450_bwd_phase1_20261017.jsonl).

`rocminfo` reports 192 CUs, four SIMDs per CU and wave32. A 192-workgroup
HW_ID1 probe consistently maps logical waves 0–7 to SIMD IDs
`0,3,2,1,0,3,2,1`: the two proposed groups each have a wave on every SIMD.
LLVM's [VGPR resource model](https://github.com/llvm/llvm-project/blob/main/llvm/lib/TargetParser/AMDGPUTargetParser.cpp)
uses 1536 wave32 register slots per SIMD (192 KiB). The earlier 512-register
behavior was not a universal compiler limit: 4-wave 32×128 and 32×256 kernels
with `waves_per_eu=1` and rematerialization disabled compile to 827 and
811 VGPRs, respectively, with zero scratch. The four-wave geometry gate passes.

The requested instruction counters were collected on the unchanged baseline
with ROCm 10.0 and with matched ROCm 10.1 libraries. Both return nonzero
`SQ_BUSY_CYCLES`/`GRBM_GUI_ACTIVE` but zero WMMA, LDS, TRANS and wait metrics,
despite the executed instructions visible in ATT. Those zero instruction
metrics are unusable; no LDS-bound conclusion or zero-percent utilization is
claimed from them. No LDS bank-conflict counter was listed.

A matched ROCm 10.1 collector/runtime resolves the ROCm 10.0 ATT collection
hang and produces a validated bundle. It profiles the frozen baseline kernel,
matching dispatch 4 (global dispatch 16), CU 0, SE mask 1, SIMD 0.
The marker is code ID 1600, code-object PC `0x40d0`, a once-per-step
`s_barrier_wait`. One step is 64 query rows; the loop is not unrolled.

| Traced WG | Median cycles/step | WMMA execution intervals / step | No-WMMA exp window | No-WMMA exp/barrier windows |
|---:|---:|---:|---:|---:|
| 0 | 8836 | 29.0% | 10.2% | 17.5% |
| 1 | 8739.5 | 29.3% | 9.4% | 17.0% |
| 2 | 8802.5 | 29.1% | 10.2% | 17.1% |

These are ATT interval measurements across both wave slots on the traced
SIMD, for zero-based steps 10–239. They use instruction latency minus stall
for WMMA execution, and subtract either wave's WMMA intervals from the
exp/barrier windows. They are not replacement values for the unavailable
SQ utilization counters. The approximately 17% common idle windows and the
measured placement satisfy the plan's gate for an experimental ping-pong path.
The selected path will still require correctness and timing evidence.

The complete viewer bundle (14 MB, kept outside the repository at
`/tmp/mi450-plan-20261017/att-baseline.tar.gz` on the measurement host)
passes JSON, SQLite, CSV and code-object validation. SHA256:
`8121d59481d3de1c7e1b31ce35e447c99f8cb8eb6fa0b3574612056f6af18684`.
The [five-step marker window](../../../benchmarks/results/mi450_bwd_att_window_20261017.json)
and [paired-wave interval analysis](../../../benchmarks/results/mi450_bwd_att_pair_analysis_20261017.json)
retain the measurement details. Baseline source SHA256 remains
`bf25907288725f4c9bac798e692c16936d10c5b6ac98d2626c7fea8708ca752c`.

### Phase 1: measurements after reboot

The GPU reset that interrupted verification required a server reboot. After
checking other users' activity and broadcasting the shutdown, the host rebooted
at 22:09 UTC. Loading the installed AMDGPU module restored all four devices;
vector, GEMM and reference checks then passed on GPU 0. The following timing
batch uses a fresh baseline from this boot, identical inputs and three
100-ms CUDA-graph trials per provider. Clocks remain unfixed. Every timing
record includes its source SHA and full configuration.

| Change | N=1024 ms | N=2048 ms |
|---|---:|---:|
| Baseline | 1.2582 | 3.8686 |
| Three Q splits | 1.1612 | 4.2917 |
| Q-panel reuse, one split | 1.1721 | 3.5983 |
| Three splits + Q reuse | 1.0832 | 4.0033 |
| Above, only sink-instructions control | 1.0871 | 4.0206 |
| Above, only reverse-local-assignment control | 1.0821 | 4.0053 |
| Above, only register-class-priority control | 1.0903 | 4.0389 |
| Above, all allocation controls disabled | 1.0893 | 4.0191 |

These rows describe source
`949891ab0b67f7272d70353d66379ed76625bf57954afb07e251ab05cfdf9a3d`.
Q splitting minimizes quantized grid waves analytically, but that alone does
not predict latency: at N=2048, the one-split reuse variant beats three splits.
The final selector therefore also uses the measured structural schedule.

One to four splits own contiguous, possibly uneven ranges in flattened
query-head/tile order. Split dK/dV accumulations use FP32 GPU-scope atomics;
preprocessing zeros both workspaces and conversion applies dK scaling once.
One split retains direct BF16 stores and five workspace buffers; multiple
splits use seven. Q reuse retains each column panel for its adjacent two dK
fragments. Loop LDS reads drop from 272 to 240 without increasing scratch;
BM64 BN256 retains 192 static WMMA instructions (160 in the steady loop).
With three splits, enabling only sink or only reverse allocation control
removes the eight-byte scratch allocation. Those controls have similar
latency to the combined legacy controls on these two shapes.

The contiguous statistics-load/register-broadcast experiment still emitted
240 loop LDS reads, so it failed the acceptance gate and was removed.
A fresh 20-call profile of three splits plus Q reuse reports main/preprocess/
conversion means of 1027.831/43.067/50.562 microseconds. The small kernels
therefore account for 8.35% of backward kernel time, excluding forward-state
and input generation. This exceeds the 5% gate for wider blocks. The filtered
[statistics CSV](../../../benchmarks/results/mi450_bwd_phase1_kernel_stats_20261017.csv)
and the Phase 1 JSONL retain the evidence.

### Phase 2: measured two-group overlap

The ATT and wave-placement gates above permit ping-pong. Waves 0–3 lead and
waves 4–7 trail, placing a wave from each group on every SIMD. The schedule
uses paired conditional barriers, two workgroup barriers per step and three
dS slots. Leading waves issue Q/dO/statistics TDM requests through disjoint
warp hints. Refilling follows the front/previous-bridge rendezvous, which
releases the trailing group's old Q/dO reads; issuing waves wait before the
bridge/front rendezvous. This establishes completion before either group
consumes the new tile.

| Candidate, three splits and Q reuse | N=1024 ms | N=2048 ms |
|---|---:|---:|
| BM64 BN256, legacy allocation controls | 0.8933 | 3.1981 |
| BM64 BN256, allocation controls disabled | 0.8712 | 3.1156 |
| BM32 BN256, legacy allocation controls | 1.0026 | 3.6310 |
| BM32 BN256, allocation controls disabled | 1.0057 | 3.6446 |

BM64 with allocation controls disabled improves on the best Phase 1 policy
by 24.3% and 15.5% in throughput, respectively. Both gains exceed the plan's
5% gate, so its conditional cross-tile fallback is not required. This
candidate uses 507 VGPRs, zero scratch, 311232 bytes of LDS, 240 loop LDS
reads and 192 static WMMA instructions. Its one-split legacy-control variant
uses 240 bytes of scratch and is excluded by the resource gate.

Four-wave BM32 geometries were measured with `waves_per_eu=1`,
rematerialization disabled, Q reuse, three splits and operand-residency
variants. They compile above 512 VGPRs without scratch but remain slower
than eight-wave ping-pong. Candidates with scratch were recorded as rejected
before GPU timing. All configurations and three-trial samples are retained
in the Phase 1 JSONL; the four-wave path remains explicitly selectable.

### Correctness fixes and coverage

GPU execution found issues that the earlier offline ordering model could not
validate. Immediate dQ requires a workgroup barrier after publishing current
dS. Split ping-pong output conversion was first worked around by preserving
wave ownership: BN256 uses a coalesced layout within each wave, while BN128
keeps native WMMA ownership. The root cause is a compiler placement bug,
described and fixed in the [18 October follow-up](#2026-10-18-follow-up). BN128's usual score layout partitions query columns
between the two wave groups; ping-pong instead partitions KV rows and uses
four full-row dK fragments so each front can complete independently.

The existing test function covers legacy controls, uneven splits, packed and
FP32 dQ, current-slot publication, outer-tile split boundaries, ping-pong at
both KV widths, and four-wave configurations. Required checks on
`(1,4,2,512)` and `(2,6,2,384)` also call backward twice to check workspace
clearing. Final source, test counts, measured selection and the ten-shape
comparison are recorded below after the final timing batch.

### Wider output conversion and selected policy

Block-size timings below use the best ping-pong policy with all allocation
controls disabled. Every entry is the median of three graph trials; the
complete 3×3 block sweep on both shapes is in the Phase 1 JSONL.

| Preprocess rows | Convert elements | N=1024 ms | N=2048 ms |
|---:|---:|---:|---:|
| 64 | 1024 | 0.8699 | 3.1185 |
| 64 | 2048 | 0.8618 | 3.0999 |
| 64 | 4096 | 0.8614 | 3.0989 |
| 128 | 4096 | 0.8653 | 3.1053 |
| 256 | 4096 | 0.8719 | 3.1185 |

Preprocessing stays at 64 rows; conversion grows from 1024 to 4096 elements.
The 2048/4096 conversion results are close, while both improve on 1024.
For 192-CU MI450 devices, GQA ratio eight, N divisible by 256 and at least
1024, and at least 256 original KV-owner workgroups, shape-aware selection
uses BM64 BN256, eight waves, three Q splits, Q reuse, ping-pong and disabled
allocation controls. Other shapes use Q reuse and the analytic split policy;
the n-only selector retains the previous full-port policy unchanged. All
older scheduling controls remain available through `BackwardConfig`.

| B | N | Original KV-owner WGs | Original quantized utilization | Analytic split count | Selected split count |
|---:|---:|---:|---:|---:|---:|
| 8 | 1024 | 256 | 66.7% | 3 | 3 |
| 8 | 2048 | 512 | 88.9% | 3 | 3 |
| 8 | 4096 | 1024 | 88.9% | 3 | 3 |
| 8 | 8192 | 2048 | 97.0% | 3 | 3 |
| 8 | 16384 | 4096 | 97.0% | 3 | 3 |
| 16 | 1024 | 512 | 88.9% | 3 | 3 |
| 16 | 2048 | 1024 | 88.9% | 3 | 3 |
| 16 | 4096 | 2048 | 97.0% | 3 | 3 |
| 16 | 8192 | 4096 | 97.0% | 3 | 3 |
| 15 | 16384 | 7680 | 100.0% | 1 | 3 |

The already full B=15 grid still selects three splits: the measured
ping-pong schedule uses that register allocation rather than the one-split
variant that fails the scratch gate. Utilization is an analytic grid-tail
calculation, not a measured WMMA utilization counter.

To reproduce the previous full-port benchmark with the new driver, override
all changed policy fields:

```sh
--config '{"q_splits":1,"reuse_q":false,"pingpong":false,"regalloc":true,"preprocess_block":64,"convert_block":1024}'
```

## 2026-10-17: final ten-shape comparison

Hq=64, Hkv=8, D=128, BF16, non-causal. Each provider receives identical
inputs and forward state. Times include preprocessing, accumulator clearing,
main kernels, conversion and AITER's GQA reductions. Values are medians of
three graph trials. The AITER column and throughput ratio use the paired
packed-path trial; FP32 trials have their own paired AITER samples in the
raw data. This is a same-device comparison on MI450.

| B | N | TLX packed ms | TLX packed TFLOP/s | TLX FP32 ms | AITER ms | AITER TFLOP/s | Packed TLX/AITER | Packed dQ NRMSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1024 | 0.859 | 800 | 1.099 | 0.997 | 689 | 1.160× | 0.353% |
| 8 | 2048 | 3.094 | 888 | 3.999 | 3.504 | 784 | 1.133× | 0.437% |
| 8 | 4096 | 11.714 | 939 | 15.219 | 13.236 | 831 | 1.130× | 0.556% |
| 8 | 8192 | 45.540 | 966 | 59.732 | 52.867 | 832 | 1.161× | 0.733% |
| 8 | 16384 | 181.666 | 968 | 236.209 | 197.439 | 891 | 1.087× | 0.993% |
| 16 | 1024 | 1.710 | 804 | 2.191 | 1.964 | 700 | 1.148× | 0.353% |
| 16 | 2048 | 6.188 | 888 | 8.004 | 6.968 | 789 | 1.126× | 0.437% |
| 16 | 4096 | 23.447 | 938 | 30.743 | 26.422 | 832 | 1.127× | 0.556% |
| 16 | 8192 | 92.072 | 955 | 119.527 | 104.981 | 838 | 1.140× | 0.734% |
| 15 | 16384 | 341.614 | 966 | 442.865 | 370.284 | 891 | 1.084× | 0.993% |

The packed policy reaches **1.084–1.161× AITER throughput**
(geometric mean 1.129×). All ten shapes meet the >=1.0 target, including
the already full B=15 grid; none falls below 0.95. The FP32 policy inherits
all scheduling changes but reaches 0.837–0.908× AITER throughput.
The packed performance gain includes its lower-precision dQ accumulation;
AITER and the FP32 column accumulate dQ in FP32. All returned gradients are
BF16. This precision difference is part of the comparison.

All 20 shape/accumulation pairs pass the 1.2% NRMSE guard for all three
gradients, including a second invocation with the same workspace. The largest
observed error, including repeats, is 0.993% NRMSE. AITER uses one
batch call except B=15,N=16384, where independent B=8 and B=7 calls avoid its
4-GiB dQ addressing defect. Both calls and the reductions are timed.

Final kernel SHA256:
`4ccdb065e28b613ec2ad2ead0015274dacd3307cdb78a90b0f86aa1d1f93ab4e`.
All 20 aligned compilations use 507 VGPRs, zero scratch, 311232 B LDS,
192 static WMMA instructions and 240 LDS reads per steady loop. Generated
IR and assembly were saved outside the repository, in
`/tmp/mi450-plan-20261017/final-source-asm/` on the measurement host.
The final source passes **106 pytest cases**, both required reference shapes
with workspace reuse, and `pre-commit run --all-files`. New untracked source,
benchmark, bridge, documentation and results also receive an explicit
`pre-commit run --files` check.

[Final three-trial timings and numerical errors](../../../benchmarks/results/mi450_bwd_final_20261017.jsonl)
contain every complete config, source hash, sample, AITER batch split and
compile result. [Per-change and verification evidence](../../../benchmarks/results/mi450_bwd_phase1_20261017.jsonl)
retain the earlier candidates, rejected experiments and resource gates.
Reproduce one final point under the documented GPU wrapper:

```sh
python benchmarks/mi450_fa_bwd.py --mode bench --shape 8,64,8,2048 \
  --aiter-library /host-tmp/mi450-aiter-bridge.so --repeat-ms 100 \
  --output /host-tmp/mi450-bwd-reproduce.jsonl
# Add --config '{"packed_dq":false}' for FP32 dQ accumulation.
```

## 2026-10-18 follow-up

### Root cause of the split ping-pong epilogue corruption

With the generic cross-wave output layout, split ping-pong returned dV with
about 10 NRMSE while dQ and dK were correct. In the TTGIR, the third
`TritonGPURemoveLayoutConversions` run rematerializes the dV conversion
directly after the `scf.for`, which places it before the closing
`amdg.cond_barrier`. Conversions carry no memory effects, so the pass may
place them anywhere before their first use. Between the two `cond_barrier`
ops, however, the two wave groups are half a step apart. The conversion's
LDS exchange inserts a workgroup barrier, and the leading group's copy of that
barrier pairs with the trailing group's end-of-step barrier. The leading
waves then read dV values that the trailing waves have not yet written.
dK was unaffected because its concatenation keeps its conversion after the
final drain.

`TritonAMDGPUSinkLayoutConversions` now pairs the `cond_barrier` ops in each
block. It moves a conversion that sits between an opening and a closing
barrier after the closing one, unless the conversion is used inside the
window. A lit case in `test/TritonGPU/amd/amd-sink-layout-conversions.mlir`
covers both the move and the in-window use. With the fix, the cross-wave
layout is correct at (1,1,1,256) with BM64 and BM32, (2,6,2,512) and
(8,64,8,1024), across three calls each. The selected kernel keeps the
wave-local layout, which needs no LDS exchange. After the compiler change it
uses 506 VGPRs and no scratch, with 192 WMMA instructions and 240 loop LDS
reads. Its timing is unchanged: 0.8577 ms and 3.0891 ms at N=1024 and 2048
against 0.859 and 3.094 ms before. All 148 AMD lit tests pass.

### Wider shape-aware selection

The 17 October selector enabled ping-pong only for 192 CUs, GQA ratio eight,
N divisible by 256 and at least 1024, and at least 256 KV-owner workgroups.
Other shapes were tested for correctness but not timed. Median of three graph
trials, packed BF16 dQ:

| B,Hq,Hkv,N | KV tile | Previous selection ms | Ping-pong ms | Splits | Change |
|---|---:|---:|---:|---:|---:|
| 8,32,8,2048 | 256 | 2.0744 | 1.6168 | 3 | +28% |
| 4,16,16,2048 | 256 | 0.6073 | 0.5282 | 3 | +15% |
| 2,64,8,1024 | 256 | 0.2838 | 0.2210 | 3 | +28% |
| 8,64,8,768 | 256 | 0.5246 | 0.5183 | 3 | +1% |
| 8,64,8,640 | 128 | 0.6196 | 0.6539 | 1 | −5% |
| 8,64,8,896 | 128 | 1.3418 | 1.1491 | 1 | +17% |
| 8,64,8,1152 | 128 | 1.8809 | 1.5883 | 1 | +18% |
| 16,64,8,1152 | 128 | 3.0775 | 2.7136 | 1 | +13% |
| 4,64,8,1152 | 128 | 0.9584 | 0.8065 | 1 | +19% |
| 4,16,16,1152 | 128 | 0.2647 | 0.2265 | 1 | +17% |
| 8,64,8,1664 | 128 | 3.8239 | 3.1841 | 1 | +20% |

The 640 row's previous selection used three splits. With one split and
no ping-pong it takes 0.5842 ms, the best measured option at that length.
At BN128, ping-pong with three splits is much slower: 2.7848 ms at
8,64,8,1152 and 1.3294 ms at 8,64,8,640. Its split partials use native,
row-scattered FP32 atomics. At BN256, one-split ping-pong spills 208 bytes
and is slower than three splits (0.6247 ms at 8,64,8,768).

The shape-aware selector therefore uses ping-pong for every BN256 shape,
with the analytic split count, or three when the analytic count is one. At
BN128 it uses one split, with ping-pong from N=896. The n-only call still
returns the previous full-port policy. All ten documented shapes select the
same configuration as before. The test suite adds ping-pong with two and four
splits; the 113 backward tests and the check shapes (1,4,2,512),
(2,6,2,384) and (8,64,8,1152) pass. Raw records are in
[the follow-up JSONL](../../../benchmarks/results/mi450_bwd_followup_20261018.jsonl).

Kernel source SHA256 after this follow-up:
`b25543109e228b8dbc4a0156d55ab63f60d7e95d46bf053f18bb2a6793bc9c52`. The ten-shape table above was
measured on the 17 October source; only the selector and comments changed.

## 2026-10-18: FP32 dQ addressing and issue schedule

This follow-up targets FP32 dQ without changing the packed policy. It starts
from commit `e30b742ed`, with the earlier layout-conversion compiler fix
already built. Fresh complete-call baselines reproduce the gap: FP32 takes
1.0983/3.9933 ms at B=8,Hq=64,Hkv=8,N=1024/2048, versus packed
0.8584/3.0883 ms. All timings below use three graph trials and the same GPU
wrapper, with device-scope atomics and unfixed clocks.

### Addressing evidence and fix

The original shuffle combines row/column terms before converting elements
to bytes. TTGIR pins these offsets to the native WMMA layout, but the
`test-print-alignment` analysis reports unit contiguity/divisibility/constancy
for both rematerialized ranges and the final offsets. LLIR has nested
`or disjoint` terms, with register constants inserted between dynamic terms;
its per-register shifts do not fold into buffer immediates. The ISA reuses
an address VGPR by rewriting it and waits for earlier atomics before reuse.

A direct logical-index rewrite alone failed the compile gate. Separating a
lane base across a whole tensor also failed because each inline-assembly
identity produced a distinct result. The accepted Python fix computes one
scalar base per hardware thread from the WMMA lane/warp bases, with additive
register deltas. A pure, empty identity prevents LLVM from mixing those
terms again. A 29-bit element mask preserves the existing wrapper's less-than-
2-GiB FP32 workspace bound and proves that adding a register delta cannot
wrap the byte offset. The permutation and output conversion are unchanged.
The largest immediate is 20224 bytes, above the plan's initial 16-KiB
estimate but within the instruction's unsigned field.

| BM64 BN256, eight waves, three splits | Original FP32 | Phase A | Phase B |
|---|---:|---:|---:|
| FP32 atomic instructions per loop | 32 | 32 | 32 |
| `s_wait_xcnt` per loop | 30 | 0 | 0 |
| Atomic clauses | two together, then serialized | one group of 32 | eight groups of four |
| Shared, unchanged address VGPR | No | Yes | Yes |
| VGPRs | 506 | 510 | 511 |
| Scratch bytes | 0 | 0 | 0 |
| Loop LDS reads | 240 | 240 | 240 |
| Static WMMA instructions | 192 | 192 | 192 |
| Workgroup barriers per loop | 2 | 2 | 2 |

The packed loop remains instruction-for-instruction identical to the fresh
baseline: 16 packed atomics in one clause, 506 VGPRs, no scratch, 240 LDS
reads and 192 static WMMAs. The benchmark now records address registers,
immediate offsets, clause sizes and the WMMA count at each atomic issue;
the compile records in the raw results below carry these fields.
`test_attention_backward_fp32_dq_addressing` asserts the FP32 loop gates.

### Spreading the FP32 atomics

Phase A leaves FP32 about 10% behind packed, triggering Phase B. The optional
`spread_dq` schedule finishes the delayed dQ at the start of the bridge and
issues its eight register slices between current dK fragments. Each slice
selects the three low register-owned row bits, so it preserves lane/warp
ownership and needs no LDS exchange. Layout pins inside the slice helper
supply its distributed argument encoding before verification.

The dS ring, TDM refill/completion protocol and conditional barriers retain
their existing phase ownership. All atomics remain in the bridge, and the
kernel still has exactly two workgroup barriers per step. The original
`amd_sched_barrier(0xff6)` matrix ordering remains; compiler-only fences
retain the VMEM groups between dK updates. The ISA issues them at WMMA
positions 32–60 in the rotated loop body, versus one burst at position 64.
There is no new dQ live range across the next front.

| FP32 schedule | B=8,N=1024 ms | B=8,N=2048 ms |
|---|---:|---:|
| Original | 1.0983 | 3.9933 |
| Constant-offset addressing | 0.9482 | 3.3967 |
| Addressing plus spreading | 0.9310 | 3.3324 |
| Paired AITER for spreading | 0.9982 | 3.5046 |

### FP32 selection sweep

The sweep covers split counts 1/2/3, ping-pong on/off and BN128/BN256,
plus spreading on the supported BN256 ping-pong variants. BN128 uses BM32;
BN256 uses BM64. All valid tile-aligned combinations were compiled: 42 were
timed with correctness guards and nine failed the scratch/resource gate.
Best complete-call medians:

| B,Hq,Hkv,N | BN | Q splits | Ping-pong | Spread dQ | FP32 ms |
|---|---:|---:|---|---|---:|
| 8,64,8,1024 | 256 | 3 | True | True | 0.9321 |
| 8,64,8,2048 | 256 | 3 | True | True | 3.3378 |
| 8,32,8,2048 | 256 | 3 | True | True | 1.7429 |
| 8,64,8,1152 | 128 | 1 | True | False | 1.7528 |

`select_config(..., packed_dq=False)` uses spreading for BN256, retaining the
measured split/tile policy. BN128 keeps the one-split policy without spreading.
The benchmark passes a `packed_dq` override to the selector before applying
other explicit controls. Literal selector tests cover all four FP32 shapes;
the packed default and n-only control behavior are retained.

[All baseline, compile, check, sweep and final records](../../../benchmarks/results/mi450_bwd_fp32dq_20261018.jsonl)
include complete configurations and source hashes. The final ten-shape audit
is recorded below.

### Final FP32 and packed comparison

Same MI450, B/Hq/Hkv/N as shown, D128 BF16 non-causal inputs. Each point
uses three graph trials for the new kernel, the saved original `e30b742ed`
kernel and AITER. Original/new order alternates across trials. Both versions
share identical inputs and output/workspace addresses; each invocation clears
its accumulators. Times include the complete backward chain. AITER uses
B=8 plus B=7 at B=15,N=16384 to avoid its known 4-GiB addressing defect.

| B | N | Original FP32 ms | New FP32 ms | AITER ms | FP32/AITER throughput | Packed ms | Packed latency change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1024 | 1.099 | 0.933 | 0.995 | 1.067× | 0.858 | -0.201% |
| 8 | 2048 | 4.001 | 3.334 | 3.502 | 1.051× | 3.090 | +0.004% |
| 8 | 4096 | 15.242 | 12.580 | 13.268 | 1.055× | 11.733 | -0.036% |
| 8 | 8192 | 59.807 | 49.294 | 52.114 | 1.057× | 45.554 | -0.002% |
| 8 | 16384 | 236.484 | 194.249 | 197.582 | 1.017× | 181.997 | -0.007% |
| 16 | 1024 | 2.191 | 1.860 | 1.963 | 1.055× | 1.708 | +0.069% |
| 16 | 2048 | 8.009 | 6.667 | 6.976 | 1.046× | 6.175 | -0.014% |
| 16 | 4096 | 30.691 | 25.404 | 26.369 | 1.038× | 23.443 | -0.018% |
| 16 | 8192 | 119.465 | 98.528 | 105.571 | 1.071× | 92.016 | +0.006% |
| 15 | 16384 | 443.473 | 364.789 | 370.907 | 1.017× | 341.681 | -0.011% |

FP32 reaches **1.017–1.071× AITER throughput** across all ten shapes,
meeting the >=1.0 target on every shape. Packed's largest measured slowdown
is **0.069%**, within the 0.3% noise allowance. Its instruction sequence
and selector policy remain unchanged. The final FP32 and packed paths both
accumulate dK/dV in FP32; the FP32 path also retains FP32 dQ accumulation.
All outputs are BF16. Atomics remain relaxed and GPU scope.

All 20 shape/precision pairs pass the 1.2% guard for dQ/dK/dV, including
workspace reuse. The largest observed NRMSE is 0.993%; FP32 dQ's
largest difference from AITER is 0.0035%. The final source passes
**125 backward tests**, including the spreading configurations and literal
FP32 selector expectations. All selected compilations stay below or at
512 VGPRs with zero scratch, zero `s_wait_xcnt`, 240 loop LDS reads,
192 static WMMAs and exactly two workgroup barriers per loop.

Measured kernel SHA256:
`9b7278dbc35bfaf39448a9441a22db199e1b5f46889dae618e7bcc4c5aa2a4bc`. The committed
source (`17c9a399f7a39b5e7e79ab7d0646670ada7acdbea868871dc88db3ffaee75bec`) only adds a compile-time
layout assertion to `_dq_register_bits`; the generated assembly is unchanged.
Original source SHA256:
`b25543109e228b8dbc4a0156d55ab63f60d7e95d46bf053f18bb2a6793bc9c52`.
[Raw results](../../../benchmarks/results/mi450_bwd_fp32dq_20261018.jsonl)
include both source hashes, full configs, all three providers' individual
samples, paired ratios, errors and compile metadata. Regenerate the assembly
with `--mode compile --artifacts <dir>`. Container checks use
`GIT_CONFIG_COUNT=1`, `GIT_CONFIG_KEY_0=safe.directory` and
`GIT_CONFIG_VALUE_0=/home/bangtliu/tlx_triton` with
`python -m pre_commit run --all-files`.

Reproduce the selected FP32 path under the documented GPU wrapper:

```sh
python benchmarks/mi450_fa_bwd.py --mode bench --shape 8,64,8,2048 \
  --config '{"packed_dq":false}' --repeat-ms 100 \
  --aiter-library /host-tmp/mi450-aiter-bridge.so \
  --output /host-tmp/mi450-fp32dq-reproduce.jsonl
```

Set `spread_dq` explicitly to false to reproduce Phase A while retaining
constant-offset FP32 addressing. The packed path remains the default.
