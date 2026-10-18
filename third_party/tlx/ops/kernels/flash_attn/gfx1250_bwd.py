"""Delayed-dQ BF16 D128 attention backward for MI450 (gfx1250).

Port of PR #2709's two-slot dS bridge to wave32 WMMA. A workgroup owns
one KV tile and loops over all Q tiles of its query-head group. Current dK
and previous dQ have independent accumulator chains. The final dQ tile is
drained explicitly. dK/dV stay resident across both Q tiles and GQA heads.
"""

import dataclasses

import torch
import triton
import triton.language as tl
import triton.language.extra.tlx as tlx

from .gfx950_bwd import _run_bwd_preprocess


@triton.constexpr_function
def _ds_shared_layout(bm, bn):
    # Match the score WMMA's per-lane Q registers so publishing dS uses
    # vector LDS stores. Its inverse view supplies the delayed-dQ transpose.
    q_bits = [1, 2, 4] + [1 << bit for bit in range(4, bm.bit_length() - 1)]
    q_bases = [[bit, 0] for bit in q_bits]
    lane_bases = [[0, bit] for bit in (1, 2, 4, 8)] + [[8, 0]]
    kv_bases = [[0, 1 << bit] for bit in range(4, bn.bit_length() - 1)]
    bases = q_bases + lane_bases + kv_bases
    return tlx.padded_shared_layout_encoding.with_bases([(512, 16)], bases, [bm, bn])


@dataclasses.dataclass(frozen=True)
class BackwardConfig:
    """Measured policy and independently selectable scheduling experiments.

    resident_kv: 0 streams K/V from LDS; 1 keeps score K/V; 2 also keeps
    the leading 3/4 of dQ's K; 3 keeps V and 7/8 of dQ's K (3/4 for BN128);
    4 keeps V alone. Physical VGPR allocation remains compiler-managed.
    outer_m stages several block_m compute phases in one Q/dO load tile.
    regalloc enables the existing three per-kernel allocation controls.
    """
    block_m: int = 64
    block_n: int = 256
    num_warps: int = 8
    use_tdm: bool = True
    stages: int = 2
    delayed: bool = True
    unroll: int = 1
    fragment_fences: bool = False
    fragment_fence_mask: int = 0
    fragmented: bool = False
    resident_kv: int = 0
    ordered: bool = False
    packed_dq: bool = False
    rematerialize: bool = False
    outer_m: int = 0
    async_stats: bool = False
    queue_wmma: bool = False
    regalloc: bool = False
    q_splits: int = 1
    waves_per_eu: int = 0
    sink_insts_to_avoid_spills: bool | None = None
    reverse_local_assignment: bool | None = None
    regclass_priority_trumps_globalness: bool | None = None
    preprocess_block: int = 64
    convert_block: int = 1024
    reuse_q: bool = False
    pingpong: bool = False
    spread_dq: bool = False


def select_config(n, b=None, hq=None, hk=None, *, num_cus=None, packed_dq=True):
    """Select the measured MI450 policy from the sequence length and grid shape.

    The n-only call preserves the one-owner policy. Shape-aware callers may
    supply num_cus for an offline/deterministic selection calculation.
    """
    config = BackwardConfig(fragmented=True, ordered=True, packed_dq=packed_dq, async_stats=True, rematerialize=True,
                            regalloc=True)
    if n % 256:
        config = dataclasses.replace(config, block_m=32, block_n=128)
    if b is None or hq is None or hk is None:
        return config
    if min(b, hq, hk) < 1 or hq % hk:
        raise ValueError("invalid batch/head counts")
    if num_cus is None:
        num_cus = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    if num_cus < 1:
        raise ValueError("num_cus must be positive")
    owners = b * hk * (n // config.block_n)
    best, best_waves = 1, triton.cdiv(owners, num_cus)
    steps = (hq // hk) * n // config.block_m
    for splits in range(2, min(4, steps) + 1):
        waves = triton.cdiv(splits * owners, num_cus)
        if waves * best < best_waves * splits:
            best, best_waves = splits, waves
    pingpong = dict(pingpong=True, regalloc=False, convert_block=4096)
    if config.block_n == 256:
        # Ping-pong was faster on every measured BN256 shape (GQA 1/4/8,
        # N 768-16384). Its one-split epilogue spills, so grids that are
        # already full still take three splits.
        # FP32 benefits from constant-offset atomics spread through the dK
        # bridge. The split/tile sweep retained the packed grid policy.
        return dataclasses.replace(config, q_splits=best if best > 1 else 3, reuse_q=True, spread_dq=not packed_dq,
                                   **pingpong)
    # BN128 ping-pong keeps native row-scattered atomics for split partials;
    # extra splits cost more than the grid tail they remove. Ping-pong was
    # slower at N=640 and faster from N=896.
    config = dataclasses.replace(config, reuse_q=True)
    return dataclasses.replace(config, **pingpong) if n >= 896 else config


def _compiler_options(config):
    options = dict(num_warps=config.num_warps, num_stages=1, enable_fp_fusion=True, waves_per_eu=config.waves_per_eu)
    for name in ("sink_insts_to_avoid_spills", "reverse_local_assignment", "regclass_priority_trumps_globalness"):
        value = getattr(config, name)
        options[name] = config.regalloc if value is None else value
    return options


@triton.jit
def _issue_tile(Q, DO, q_desc, do_desc, qb, dob, step, N: tl.constexpr, BM: tl.constexpr, STAGES: tl.constexpr,
                TDM: tl.constexpr, lse_desc=None, delta_desc=None, lseb=None, db=None,
                ASYNC_STATS: tl.constexpr = False, PINGPONG: tl.constexpr = False):
    slot = step % STAGES
    if PINGPONG:
        # Waves 0..3 own every TDM request. With the half-step phase shift,
        # the trailing group has not reached its next issue when the leading
        # group starts consuming that tile. Disjoint hints split the two
        # requests within the leading group, keeping their completion local.
        q_next = tlx.update_tensor_descriptor(q_desc, add_offsets=[step * BM, 0], clamp_bounds=False)
        do_next = tlx.update_tensor_descriptor(do_desc, add_offsets=[step * BM, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_load_fused([(q_next, tlx.local_view(qb, slot), 0b0011),
                                             (do_next, tlx.local_view(dob, slot), 0b1100)])
    elif TDM:
        tlx.async_amd_descriptor_load(q_desc, tlx.local_view(qb, slot), [step * BM, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_load(do_desc, tlx.local_view(dob, slot), [step * BM, 0], clamp_bounds=False)
    else:
        offsets = (step * BM + tl.arange(0, BM))[:, None] * 128 + tl.arange(0, 128)[None, :]
        tlx.async_load(Q + offsets, tlx.local_view(qb, slot))
        tlx.async_load(DO + offsets, tlx.local_view(dob, slot))
        tlx.async_load_commit_group()
    if ASYNC_STATS:
        if PINGPONG:
            lse_next = tlx.update_tensor_descriptor(lse_desc, add_offsets=[0, step * BM], clamp_bounds=False)
            delta_next = tlx.update_tensor_descriptor(delta_desc, add_offsets=[0, step * BM], clamp_bounds=False)
            tlx.async_amd_descriptor_load_fused([(lse_next, tlx.local_view(lseb, slot), 0b0001),
                                                 (delta_next, tlx.local_view(db, slot), 0b0010)])
        else:
            tlx.async_amd_descriptor_load(lse_desc, tlx.local_view(lseb, slot), [0, step * BM], clamp_bounds=False)
            tlx.async_amd_descriptor_load(delta_desc, tlx.local_view(db, slot), [0, step * BM], clamp_bounds=False)


@triton.jit
def _wmma(a, b, acc, ORDERED: tl.constexpr):
    # Matrix-only compiler scheduling boundaries preserve native WMMA
    # alternation while allowing LDS/global loads to overlap the fragments.
    # No workgroup barrier or hardware completion wait is emitted here.
    if ORDERED:
        tlx.amd_sched_barrier(0xFF6)
    result = tl.dot(a, b, acc)
    if ORDERED:
        tlx.amd_sched_barrier(0xFF6)
    return result


@triton.constexpr_function
def _dq_register_bits(bm, mma):
    row_warps = sum(16 * basis[0] for basis in mma.warp_bases)
    col_warps = sum(16 * basis[1] for basis in mma.warp_bases)
    # The per-thread base swaps column 16 with lane-owned row 8. A register-
    # owned column 16 would need the swap in its delta, so reject it.
    assert col_warps & 16, "FP32 dQ addressing requires warp-owned column 16"
    return ((bm - 1) & ~(8 | row_warps)) * 128 + (127 & ~(15 | col_warps))


@triton.jit
def _dq_offsets(step, BM: tl.constexpr, MMA: tl.constexpr, PACKED: tl.constexpr = False, REMAT: tl.constexpr = False):
    if REMAT:
        local_rows = tlx.rematerialized_range(0, BM, 27090, placement=step)
        cols = tlx.rematerialized_range(0, 128, 27091, placement=step)[None, :]
    else:
        local_rows = tl.arange(0, BM)
        cols = tl.arange(0, 128)[None, :]
    rows = (step * BM + local_rows)[:, None]
    # Swap the row-8 and column-16 bits inside each 16x32 patch. The
    # WMMA's lane-16 bit then addresses the adjacent 16 columns, giving
    # each wave a contiguous 128-byte atomic segment (aiter's shuffle).
    if PACKED:
        # Native WMMA registers advance in rows. Make row bit 0 the
        # contiguous BF16 pair, followed by the five lane bits. Merely
        # declaring contiguity=2 on the FP32 shuffle produces misaligned
        # packed atomics because those registers address different rows.
        logical = rows * 128 + cols
        offsets = ((logical & ~(15 | 16 | 32 | 128 | 1024)) | ((logical & 15) << 1) | ((logical & 16) << 6) |
                   ((logical & 32) << 2) | ((logical & 128) >> 7) | ((logical & 1024) >> 5))
    else:
        # Swap bits 4 (column 16) and 10 (row 8), separating the register
        # coordinates from the lane/warp base before the permutation.
        register_bits: tl.constexpr = _dq_register_bits(BM, MMA)
        logical = rows * 128 + cols
        thread = tlx.thread_id(0)
        row_lane = (thread & 16) >> 1
        col_lane = thread & 15
        for warp_bit in tl.static_range(len(MMA.warp_bases)):
            warp = (thread >> (5 + warp_bit)) & 1
            row_lane += warp * (16 * MMA.warp_bases[warp_bit][0])
            col_lane += warp * (16 * MMA.warp_bases[warp_bit][1])
        lane_rows = step * BM + (row_lane & (BM - 1))
        base = ((lane_rows & ~8) * 128 + ((col_lane & 16) << 6) + (col_lane & ~16) + ((lane_rows & 8) << 1))
        # Keep LLVM from reassociating the dynamic terms through each
        # register's constant delta. One scalar identity emits no instruction
        # and supplies the same base to every register owned by this thread.
        base = tl.inline_asm_elementwise("", "=v,0", [base], dtype=tl.int32, is_pure=True, pack=1)
        # fa_backward bounds each query-head group's FP32 workspace below
        # 2 GiB. Retain that range fact across the opaque identity so adding
        # the register delta cannot wrap the 32-bit byte offset.
        base = base & ((1 << 29) - 1)
        offsets = base + (logical & register_bits)
    offsets = tlx.require_layout(offsets, MMA)
    return offsets


@triton.jit
def _dq_atomic(dq, DQA, step, SM_SCALE: tl.constexpr, BM: tl.constexpr, MMA: tl.constexpr, PACKED: tl.constexpr = False,
               REMAT: tl.constexpr = False):
    offsets = _dq_offsets(step, BM, MMA, PACKED, REMAT)
    dq = tlx.require_layout(dq * SM_SCALE, MMA)
    if PACKED:
        tlx.buffer_atomic_add(DQA, offsets, dq.to(tl.bfloat16), sem="relaxed", scope="gpu", contiguity=2)
    else:
        tlx.buffer_atomic_add(DQA, offsets, dq, sem="relaxed", scope="gpu")


@triton.jit
def _dq_atomic_group(dq_groups, offset_groups, DQA, GROUP: tl.constexpr, BM: tl.constexpr):
    # The first dimension is the three low row bits, all register-owned.
    # Slicing it leaves every lane/warp active and emits four FP32 atomics
    # per thread at BM64 (two at BM32), without an LDS layout conversion.
    group_layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2, 2), (8, BM // 32, 2)),
                                            stride=((BM // 8, 1, 2 * BM, 4 * BM, 2), (16 * BM, 4, 8 * BM)))
    dq_groups = tlx.require_layout(dq_groups, group_layout)
    offset_groups = tlx.require_layout(offset_groups, group_layout)
    dq = tlx.extract_slice(dq_groups, [1, BM // 8 * 128], [GROUP, 0])
    offsets = tlx.extract_slice(offset_groups, [1, BM // 8 * 128], [GROUP, 0])
    tlx.buffer_atomic_add(DQA, offsets, dq, sem="relaxed", scope="gpu")


@triton.jit
def _delayed_bwd_kernel(Q, K, V, DO, LSE, Delta, DQA, DK, DV, HQ: tl.constexpr, HK: tl.constexpr, N: tl.constexpr,
                        SM_SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, NW: tl.constexpr, TDM: tl.constexpr,
                        STAGES: tl.constexpr, DELAYED: tl.constexpr, UNROLL: tl.constexpr,
                        FRAGMENT_FENCES: tl.constexpr = False, FRAGMENT_FENCE_MASK: tl.constexpr = 0,
                        FRAGMENTED: tl.constexpr = False, RESIDENT_KV: tl.constexpr = 0, ORDERED: tl.constexpr = False,
                        PACKED_DQ: tl.constexpr = False, REMATERIALIZE: tl.constexpr = False, OUTER_M: tl.constexpr = 0,
                        ASYNC_STATS: tl.constexpr = False, QUEUE_WMMA: tl.constexpr = False, Q_SPLITS: tl.constexpr = 1,
                        REUSE_Q: tl.constexpr = False, PINGPONG: tl.constexpr = False, SPREAD_DQ: tl.constexpr = False):
    if QUEUE_WMMA:
        tlx.amd_set_wave_sched_mode(1, offset=2, width=1)
    kv_head = tl.program_id(0)
    kv_tile = tl.program_id(1)
    batch_split = tl.program_id(2)
    batch = batch_split // Q_SPLITS
    split = batch_split % Q_SPLITS
    GROUP: tl.constexpr = HQ // HK
    OM: tl.constexpr = OUTER_M if OUTER_M else BM
    PHASES: tl.constexpr = OM // BM
    DS_SLOTS: tl.constexpr = 3 if PINGPONG else 2
    # At BN128 the usual fragmented layout partitions query columns across
    # wave groups. Ping-pong instead partitions KV rows and keeps four full
    # row fragments, so either group can finish its front independently.
    PAIRED_FRAGMENTS: tl.constexpr = FRAGMENTED and (not PINGPONG or BN == 256)
    TDM_PER_STAGE: tl.constexpr = 4 if ASYNC_STATS else 2
    steps: tl.constexpr = GROUP * N // BM
    step_start = split * steps // Q_SPLITS
    step_stop = (split + 1) * steps // Q_SPLITS
    outer_start = step_start // PHASES
    outer_stop = tl.cdiv(step_stop, PHASES)
    q_base = (batch.to(tl.int64) * HQ + kv_head * GROUP) * N * 128
    kv_base = (batch.to(tl.int64) * HK + kv_head) * N * 128
    stats_base = (batch.to(tl.int64) * HQ + kv_head * GROUP) * N
    Q += q_base
    DO += q_base
    DQA += q_base
    K += kv_base
    V += kv_base
    DK += kv_base
    DV += kv_base
    LSE += stats_base
    Delta += stats_base

    if NW == 16:
        if BN == 256:
            kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (4, 0), (0, 1)))
        else:
            kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (0, 1), (0, 2)))
        dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (0, 2), (1, 0), (2, 0)), transposed=False)
    elif NW == 4:
        kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0)))
        dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (0, 2)), transposed=False)
    elif NW == 8:
        if FRAGMENTED and BN == 128 and not PINGPONG:
            kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (0, 1)))
        else:
            kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), (2, 0), (4, 0)))
        dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), (0, 2), (1, 0)), transposed=False)
    else:
        kv_mma: tl.constexpr = tlx.amd_wmma_layout(((1, 0), ))
        dq_mma: tl.constexpr = tlx.amd_wmma_layout(((0, 1), ), transposed=False)
    lhs: tl.constexpr = tlx.dot_operand_layout(0, kv_mma, k_width=8)
    rhs: tl.constexpr = tlx.dot_operand_layout(1, kv_mma, k_width=8)
    dq_lhs: tl.constexpr = tlx.dot_operand_layout(0, dq_mma, k_width=8)
    dq_rhs: tl.constexpr = tlx.dot_operand_layout(1, dq_mma, k_width=8)
    kb = tlx.local_alloc((BN, 128), tl.bfloat16, 1)
    vb = tlx.local_alloc((BN, 128), tl.bfloat16, 1)
    qb = tlx.local_alloc((OM, 128), tl.bfloat16, STAGES)
    dob = tlx.local_alloc((OM, 128), tl.bfloat16, STAGES)
    ds_layout: tl.constexpr = _ds_shared_layout(BM, BN)
    dsb = tlx.local_alloc((BM, BN), tl.bfloat16, DS_SLOTS, layout=ds_layout)
    if DELAYED:
        # Seed the previous slot: phase zero can execute the same bridge as
        # every later phase, publishing a harmless zero dQ into tile zero.
        tlx.local_store(tlx.local_view(dsb, (step_start + DS_SLOTS - 1) % DS_SLOTS), tl.zeros((BM, BN), tl.bfloat16))

    lse_desc = tl.make_tensor_descriptor(LSE, [1, GROUP * N], [GROUP * N, 1], [1, OM])
    delta_desc = tl.make_tensor_descriptor(Delta, [1, GROUP * N], [GROUP * N, 1], [1, OM])
    lseb = tlx.local_alloc((1, OM), tl.float32, STAGES)
    db = tlx.local_alloc((1, OM), tl.float32, STAGES)
    q_desc = tl.make_tensor_descriptor(Q, [GROUP * N, 128], [128, 1], [OM, 128])
    do_desc = tl.make_tensor_descriptor(DO, [GROUP * N, 128], [128, 1], [OM, 128])
    if TDM:
        k_desc = tl.make_tensor_descriptor(K, [N, 128], [128, 1], [BN, 128])
        v_desc = tl.make_tensor_descriptor(V, [N, 128], [128, 1], [BN, 128])
        tlx.async_amd_descriptor_load(k_desc, tlx.local_view(kb, 0), [kv_tile * BN, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_load(v_desc, tlx.local_view(vb, 0), [kv_tile * BN, 0], clamp_bounds=False)
        tlx.async_amd_descriptor_wait(0)
    else:
        kv_offsets = (kv_tile * BN + tl.arange(0, BN))[:, None] * 128 + tl.arange(0, 128)[None, :]
        tlx.async_load(K + kv_offsets, tlx.local_view(kb, 0))
        tlx.async_load(V + kv_offsets, tlx.local_view(vb, 0))
        tlx.async_load_commit_group()
        tlx.async_load_wait_group(0)

    if (RESIDENT_KV == 1 or RESIDENT_KV == 2):
        k_resident = tlx.local_load(tlx.local_view(kb, 0), layout=lhs)
    if RESIDENT_KV:
        v_resident = tlx.local_load(tlx.local_view(vb, 0), layout=lhs)
    if (RESIDENT_KV == 2 or RESIDENT_KV == 3):
        # Keep the reused leading 3/4 of K in delayed-dQ ownership. The last
        # quarter remains a streamed LDS operand, limiting the resident bank.
        k_dq_lo = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [0, 0], [BN // 2, 128]), layout=dq_rhs)
        k_dq_mid = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [BN // 2, 0], [BN // 4, 128]), layout=dq_rhs)
    if RESIDENT_KV == 3 and BN == 256:
        k_dq_band6 = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [192, 0], [32, 128]), layout=dq_rhs)

    dk0 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
    dk1 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
    dk2 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
    dk3 = tlx.zeros((BN, 32), tl.float32, layout=kv_mma)
    if PAIRED_FRAGMENTS:
        dks = (tlx.zeros((BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros(
            (BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros(
                (BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros((BN // 2, 32), tl.float32, layout=kv_mma),
               tlx.zeros((BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros(
                   (BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros(
                       (BN // 2, 32), tl.float32, layout=kv_mma), tlx.zeros((BN // 2, 32), tl.float32, layout=kv_mma))
    dv = tlx.zeros((BN, 128), tl.float32, layout=kv_mma)
    for initial in tl.static_range(STAGES - 1):
        if Q_SPLITS == 1 or outer_start + initial < outer_stop:
            _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer_start + initial, N, OM, STAGES, TDM, lse_desc,
                        delta_desc, lseb, db, ASYNC_STATS, PINGPONG)

    if PINGPONG:
        tlx.async_amd_descriptor_wait(0)
        tlx.workgroup_barrier()
        # Measured wave placement puts waves w and w+4 on the same SIMD.
        group_b = tlx.thread_id(0) >= 128
        tlx.cond_barrier(group_b)

    for outer in tl.range(outer_start, outer_stop, loop_unroll_factor=UNROLL):
        # The next tile is in flight while this tile computes its five GEMMs.
        if not PINGPONG and outer + STAGES - 1 < outer_stop:
            _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer + STAGES - 1, N, OM, STAGES, TDM, lse_desc, delta_desc,
                        lseb, db, ASYNC_STATS)
        if PINGPONG:
            pass
        elif TDM:
            tlx.async_amd_descriptor_wait(TDM_PER_STAGE * (STAGES - 1))
        else:
            tlx.async_load_wait_group(STAGES - 1)
        # The last iterations have fewer outstanding copies. A fixed nonzero
        # wait count cannot establish readiness for those draining tiles.
        if not PINGPONG and outer + STAGES - 1 >= outer_stop:
            if TDM:
                tlx.async_amd_descriptor_wait(0)
            else:
                tlx.async_load_wait_group(0)
        for phase in tl.static_range(PHASES):
            step = outer * PHASES + phase
            if Q_SPLITS == 1 or PHASES == 1 or (step >= step_start and step < step_stop):
                qv = tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, 0], [BM, 128])
                dov = tlx.local_slice(tlx.local_view(dob, outer % STAGES), [phase * BM, 0], [BM, 128])
                if (RESIDENT_KV == 1 or RESIDENT_KV == 2):
                    k = k_resident
                else:
                    k = tlx.local_load(tlx.local_view(kb, 0), layout=lhs, relaxed=True)
                qt = tlx.local_load(tlx.local_trans(qv), layout=rhs, relaxed=True)
                scores = tl.dot(k, qt, tlx.zeros((BN, BM), tl.float32, layout=kv_mma))
                if RESIDENT_KV:
                    v = v_resident
                else:
                    v = tlx.local_load(tlx.local_view(vb, 0), layout=lhs, relaxed=True)
                dot = tlx.local_load(tlx.local_trans(dov), layout=rhs, relaxed=True)
                dp = tl.dot(v, dot, tlx.zeros((BN, BM), tl.float32, layout=kv_mma))
                rows = step * BM + tl.arange(0, BM)
                if ASYNC_STATS:
                    lse = tl.reshape(
                        tlx.local_load(tlx.local_slice(tlx.local_view(lseb, outer % STAGES), [0, phase * BM], [1, BM])),
                        (BM, ))
                    delta = tl.reshape(
                        tlx.local_load(tlx.local_slice(tlx.local_view(db, outer % STAGES), [0, phase * BM], [1, BM])),
                        (BM, ))
                else:
                    lse = tl.load(LSE + rows)
                    delta = tl.load(Delta + rows)
                p = tl.exp2(scores * (SM_SCALE * 1.4426950408889634) - lse[None, :] * 1.4426950408889634)
                ds = (p * (dp - delta[None, :])).to(tl.bfloat16)
                tlx.local_store(tlx.local_view(dsb, step % DS_SLOTS), tl.trans(ds))
                p = tlx.require_layout(p.to(tl.bfloat16), lhs, pin=False)
                do_out = tlx.local_load(dov, layout=rhs, relaxed=True)
                dv = tl.dot(p, do_out, dv)
                ds = tlx.require_layout(ds, lhs, pin=False)
                if PINGPONG:
                    # Leading front(i) rendezvous with trailing bridge(i-1).
                    # Its last Q/dO reads are complete before slot i+1 reuses
                    # the two-stage ring's slot i-1. Only leading waves issue
                    # the refill; their wait precedes the next rendezvous.
                    tlx.workgroup_barrier()
                    if outer + STAGES - 1 < outer_stop:
                        _issue_tile(Q, DO, q_desc, do_desc, qb, dob, outer + STAGES - 1, N, OM, STAGES, TDM, lse_desc,
                                    delta_desc, lseb, db, ASYNC_STATS, PINGPONG)
                if not DELAYED:
                    # Immediate dQ consumes the publication from this phase
                    # across all waves. The delayed path reads the other slot,
                    # whose publication is covered by the preceding end barrier.
                    tlx.workgroup_barrier()

                if PAIRED_FRAGMENTS:
                    if DELAYED:
                        dq_step = tl.maximum(step - 1, step_start)
                        dq_slot = (step + DS_SLOTS - 1) % DS_SLOTS
                    else:
                        dq_step = step
                        dq_slot = step % DS_SLOTS
                    previous = tlx.local_view(dsb, dq_slot)
                    if SPREAD_DQ:
                        # Finish the previous dQ before the current dK chain.
                        # Its eight register slices can then issue between dK
                        # fragments without carrying dQ into the next front.
                        dq0 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                        dq1 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                        ds_band = tlx.local_load(tlx.local_slice(previous, [0, 0], [BM, 32]), layout=dq_lhs,
                                                 relaxed=True)
                        for band in tl.static_range(8):
                            if band < 7:
                                next_ds = tlx.local_load(tlx.local_slice(previous, [0, (band + 1) * 32], [BM, 32]),
                                                         layout=dq_lhs, relaxed=True)
                            k0 = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [band * 32, 0], [32, 64]),
                                                layout=dq_rhs, relaxed=True)
                            k1 = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [band * 32, 64], [32, 64]),
                                                layout=dq_rhs, relaxed=True)
                            dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                            dq1 = _wmma(ds_band, k1, dq1, ORDERED)
                            if band < 7:
                                ds_band = next_ds
                        atomic_layout: tl.constexpr = tlx.layout(shape=((16, 2, 2, 2, 2), (8, BM // 32, 2)),
                                                                 stride=((1, 1024, 16, 32, 2048), (128, 4096, 64)))
                        dq = tlx.require_layout(tl.cat(dq0, dq1, dim=1) * SM_SCALE, atomic_layout)
                        offsets = tlx.require_layout(_dq_offsets(dq_step, BM, dq_mma, False, REMATERIALIZE),
                                                     atomic_layout)
                        dq_groups = tl.trans(tl.reshape(tl.trans(dq), (128 * BM // 8, 8)))
                        offset_groups = tl.trans(tl.reshape(tl.trans(offsets), (128 * BM // 8, 8)))
                        for band in tl.static_range(8):
                            ds_current = tlx.extract_slice(ds, [BN // 2, BM], [(band % 2) * (BN // 2), 0])
                            if not REUSE_Q or band % 2 == 0:
                                q_part = tlx.local_load(
                                    tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, (band // 2) * 32],
                                                    [BM, 32]), layout=rhs, relaxed=True)
                            updated = _wmma(ds_current, q_part, dks[band], ORDERED)
                            dks = dks[:band] + (updated, ) + dks[band + 1:]
                            _dq_atomic_group(dq_groups, offset_groups, DQA, band, BM)
                            # Compiler-only fence retains each VMEM group next
                            # to its dK fragment; it emits no workgroup barrier.
                            tlx.amd_sched_barrier(0)
                    else:
                        # Start delayed-dQ reads before dK, then consume one K=32 band
                        # between native dK fragments. Independent head-column halves
                        # supply two dQ accumulator chains.
                        ds_band = tlx.local_load(tlx.local_slice(previous, [0, 0], [BM, 32]), layout=dq_lhs,
                                                 relaxed=True)
                        dq0 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                        dq1 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
                        for band in tl.static_range(8):
                            ds_current = tlx.extract_slice(ds, [BN // 2, BM], [(band % 2) * (BN // 2), 0])
                            if not REUSE_Q or band % 2 == 0:
                                # The adjacent low/high KV fragments consume the
                                # same Q panel. Retain it across their dQ band.
                                q_part = tlx.local_load(
                                    tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, (band // 2) * 32],
                                                    [BM, 32]), layout=rhs, relaxed=True)
                            if band < BN // 32:
                                if band + 1 < BN // 32:
                                    next_ds = tlx.local_load(tlx.local_slice(previous, [0, (band + 1) * 32], [BM, 32]),
                                                             layout=dq_lhs, relaxed=True)
                                if (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < BN // 64:
                                    k0 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 0])
                                    k1 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 64])
                                elif (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < 3 * BN // 128:
                                    k0 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 0])
                                    k1 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 64])
                                elif RESIDENT_KV == 3 and BN == 256 and band == 6:
                                    k0 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 0])
                                    k1 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 64])
                                else:
                                    k0 = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(kb, 0), [band * 32, 0], [32, 64]), layout=dq_rhs,
                                        relaxed=True)
                                    k1 = tlx.local_load(
                                        tlx.local_slice(tlx.local_view(kb, 0), [band * 32, 64], [32, 64]),
                                        layout=dq_rhs, relaxed=True)
                            updated = _wmma(ds_current, q_part, dks[band], ORDERED)
                            dks = dks[:band] + (updated, ) + dks[band + 1:]
                            if band < BN // 32:
                                dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                                dq1 = _wmma(ds_band, k1, dq1, ORDERED)
                                if band + 1 < BN // 32:
                                    ds_band = next_ds
                        dq = tl.cat(dq0, dq1, dim=1)
                        _dq_atomic(dq, DQA, dq_step, SM_SCALE, BM, dq_mma, PACKED_DQ, REMATERIALIZE)
                else:
                    dq = tlx.zeros((BM, 128), tl.float32, layout=dq_mma)
                    # Fragment current dK and delayed dQ so both WMMA chains are visible
                    # to LLVM. dK consumes registers; dQ reads the other dS LDS slot.
                    for band in tl.static_range(4):
                        q_part = tlx.local_load(
                            tlx.local_slice(tlx.local_view(qb, outer % STAGES), [phase * BM, band * 32], [BM, 32]),
                            layout=rhs, relaxed=True)
                        if band == 0:
                            dk0 = tl.dot(ds, q_part, dk0)
                        elif band == 1:
                            dk1 = tl.dot(ds, q_part, dk1)
                        elif band == 2:
                            dk2 = tl.dot(ds, q_part, dk2)
                        else:
                            dk3 = tl.dot(ds, q_part, dk3)
                        if FRAGMENT_FENCES:
                            tlx.amd_sched_barrier(FRAGMENT_FENCE_MASK)
                        if DELAYED:
                            dq_step = tl.maximum(step - 1, step_start)
                            dq_slot = (step + DS_SLOTS - 1) % DS_SLOTS
                        else:
                            dq_step = step
                            dq_slot = step % DS_SLOTS
                        # BN/4 is at least the native reduction width (32).
                        ds_part = tlx.local_load(
                            tlx.local_slice(tlx.local_view(dsb, dq_slot), [0, band * (BN // 4)], [BM, BN // 4]),
                            layout=dq_lhs, relaxed=True)
                        k_part = tlx.local_load(
                            tlx.local_slice(tlx.local_view(kb, 0), [band * (BN // 4), 0], [BN // 4, 128]),
                            layout=dq_rhs, relaxed=True)
                        dq = tl.dot(ds_part, k_part, dq)
                        if FRAGMENT_FENCES:
                            tlx.amd_sched_barrier(FRAGMENT_FENCE_MASK)
                    _dq_atomic(dq, DQA, dq_step, SM_SCALE, BM, dq_mma, PACKED_DQ, REMATERIALIZE)
                if PINGPONG:
                    tlx.async_amd_descriptor_wait(0)
                    # Leading bridge(i) rendezvous with trailing front(i).
                    tlx.workgroup_barrier()
                else:
                    tl.debug_barrier()

    if PINGPONG:
        tlx.cond_barrier(~group_b)

    if DELAYED:
        if PAIRED_FRAGMENTS:
            dq0 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
            dq1 = tlx.zeros((BM, 64), tl.float32, layout=dq_mma)
            for band in tl.static_range(BN // 32):
                ds_band = tlx.local_load(
                    tlx.local_slice(tlx.local_view(dsb, (step_stop - 1) % DS_SLOTS), [0, band * 32], [BM, 32]),
                    layout=dq_lhs)
                if (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < BN // 64:
                    k0 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 0])
                    k1 = tlx.extract_slice(k_dq_lo, [32, 64], [band * 32, 64])
                elif (RESIDENT_KV == 2 or RESIDENT_KV == 3) and band < 3 * BN // 128:
                    k0 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 0])
                    k1 = tlx.extract_slice(k_dq_mid, [32, 64], [band * 32 - BN // 2, 64])
                elif RESIDENT_KV == 3 and BN == 256 and band == 6:
                    k0 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 0])
                    k1 = tlx.extract_slice(k_dq_band6, [32, 64], [0, 64])
                else:
                    k0 = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [band * 32, 0], [32, 64]), layout=dq_rhs,
                                        relaxed=True)
                    k1 = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [band * 32, 64], [32, 64]),
                                        layout=dq_rhs, relaxed=True)
                dq0 = _wmma(ds_band, k0, dq0, ORDERED)
                dq1 = _wmma(ds_band, k1, dq1, ORDERED)
            dq = tl.cat(dq0, dq1, dim=1)
        else:
            dq = tlx.zeros((BM, 128), tl.float32, layout=dq_mma)
            for band in tl.static_range(4):
                ds_part = tlx.local_load(
                    tlx.local_slice(tlx.local_view(dsb, (step_stop - 1) % DS_SLOTS), [0, band * (BN // 4)],
                                    [BM, BN // 4]), layout=dq_lhs)
                k_part = tlx.local_load(tlx.local_slice(tlx.local_view(kb, 0), [band * (BN // 4), 0], [BN // 4, 128]),
                                        layout=dq_rhs)
                dq = tl.dot(ds_part, k_part, dq)
        _dq_atomic(dq, DQA, step_stop - 1, SM_SCALE, BM, dq_mma, PACKED_DQ, REMATERIALIZE)
    if PAIRED_FRAGMENTS:
        dk0 = tl.cat(dks[0], dks[1], dim=0)
        dk1 = tl.cat(dks[2], dks[3], dim=0)
        dk2 = tl.cat(dks[4], dks[5], dim=0)
        dk3 = tl.cat(dks[6], dks[7], dim=0)
    dk = tl.cat(tl.cat(dk0, dk1, dim=1), tl.cat(dk2, dk3, dim=1), dim=1)
    offsets = (kv_tile * BN + tl.arange(0, BN))[:, None] * 128 + tl.arange(0, 128)[None, :]
    if Q_SPLITS > 1:
        # Native KV WMMA lanes advance in rows. Redistribute once in the
        # epilogue so each FP32 atomic instruction writes a contiguous row
        # segment, instead of scattering its lanes across 128-column strides.
        if PINGPONG and BN == 128:
            # Full-row fragments retain native ownership at this width.
            output_layout: tl.constexpr = kv_mma
        elif PINGPONG:
            # Preserve wave ownership after the phase-shifted loop, avoiding
            # an LDS exchange. Each wave owns two groups of 16 KV rows at BN256.
            # (Cross-wave conversions are also correct now that the AMD sink
            # pass moves them past the closing cond_barrier.)
            output_layout: tl.constexpr = tlx.layout(shape=((32, NW), (4, 16, BN // (16 * NW))),
                                                     stride=((1, 16 * 128), (32, 128, 128 * 128)))
        else:
            output_layout: tl.constexpr = tlx.layout(shape=((32, NW), (4, BN // NW)), stride=((1, 128), (32, 128 * NW)))
        offsets = tlx.require_layout(offsets, output_layout)
        dk = tlx.require_layout(dk, output_layout)
        dv = tlx.require_layout(dv, output_layout)
        tlx.buffer_atomic_add(DK, offsets, dk, sem="relaxed", scope="gpu")
        tlx.buffer_atomic_add(DV, offsets, dv, sem="relaxed", scope="gpu")
    else:
        tl.store(DK + offsets, dk * SM_SCALE)
        tl.store(DV + offsets, dv)


@triton.jit
def _convert_outputs(DQA, DQ, DKA, DVA, DK, DV, SIZE: tl.constexpr, KV_SIZE: tl.constexpr, SM_SCALE: tl.constexpr,
                     BLOCK: tl.constexpr, PACKED: tl.constexpr = False, PARTIALS: tl.constexpr = False):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    shuffled = ((offsets & ~(1024 | 16)) | ((offsets & 1024) >> 6) | ((offsets & 16) << 6))
    if PACKED:
        shuffled = ((offsets & ~(15 | 16 | 32 | 128 | 1024)) | ((offsets & 15) << 1) | ((offsets & 16) << 6) |
                    ((offsets & 32) << 2) | ((offsets & 128) >> 7) | ((offsets & 1024) >> 5))
    tl.store(DQ + offsets, tl.load(DQA + shuffled, offsets < SIZE, 0), offsets < SIZE)
    if PARTIALS:
        dk = tl.load(DKA + offsets, offsets < KV_SIZE, 0)
        dv = tl.load(DVA + offsets, offsets < KV_SIZE, 0)
        tl.store(DK + offsets, dk * SM_SCALE, offsets < KV_SIZE)
        tl.store(DV + offsets, dv, offsets < KV_SIZE)


def allocate_workspace(q, k, *, config=None, packed_dq=None):
    """Allocate the five legacy buffers and optional FP32 dK/dV partials."""
    config = config or select_config(q.shape[-2], q.shape[0], q.shape[1], k.shape[1],
                                     packed_dq=True if packed_dq is None else packed_dq)
    if packed_dq is None:
        packed_dq = config.packed_dq
    workspace = (torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32),
                 torch.empty(q.shape, device=q.device, dtype=torch.bfloat16 if packed_dq else torch.float32),
                 torch.empty_like(q), torch.empty_like(k), torch.empty_like(k))
    if config.q_splits > 1:
        workspace += (torch.empty_like(k, dtype=torch.float32), torch.empty_like(k, dtype=torch.float32))
    return workspace


def fa_backward(q, k, v, o, do, lse, sm_scale=None, causal=False, *, config=None, workspace=None):
    """Return dQ/dK/dV for contiguous BHND BF16 inputs, equal lengths, D128."""
    b, hq, n, d = q.shape
    hk = k.shape[1]
    if min(b, hq, hk) < 1 or hq % hk:
        raise ValueError("invalid batch/head counts")
    if q.device.type != "cuda":
        raise ValueError("inputs must be on a GPU")
    config = config or select_config(n, b, hq, hk)
    if config.block_n not in (128, 256) or config.block_m not in (16, 32, 64) or config.num_warps not in (2, 4, 8, 16):
        raise ValueError("unsupported WMMA tile or wave count")
    if causal or d != 128 or n < 256 or n % config.block_n or n % config.block_m:
        raise ValueError("gfx1250 backward requires non-causal D128 and tile-aligned N >= 256")
    if config.stages not in (2, 3, 4) or config.unroll not in (1, 2, 4, 8):
        raise ValueError("unsupported pipeline depth or unroll")
    if not isinstance(config.fragment_fence_mask, int) or not 0 <= config.fragment_fence_mask <= 0xFFF:
        raise ValueError("fragment scheduling mask must be an integer in [0, 0xfff]")
    outer_m = config.outer_m or config.block_m
    if outer_m not in (16, 32, 64, 128) or outer_m < config.block_m or n % outer_m:
        raise ValueError("outer Q tile must be an aligned multiple of the compute tile")
    if config.async_stats and not config.use_tdm:
        raise ValueError("asynchronous statistics require TDM")
    if config.pingpong and (not config.use_tdm or not config.delayed or not config.fragmented or config.num_warps != 8
                            or config.stages != 2 or outer_m != config.block_m):
        raise ValueError("ping-pong requires delayed fragmented TDM, eight waves, two stages and one compute phase")
    if config.spread_dq and (config.packed_dq or not config.pingpong or not config.fragmented or not config.delayed
                             or config.num_warps != 8 or config.block_n != 256 or config.block_m not in (32, 64)
                             or config.resident_kv != 0):
        raise ValueError("spread dQ requires FP32 ping-pong, eight waves, BM32/64 BN256 and streamed KV")
    if config.resident_kv not in (0, 1, 2, 3, 4):
        raise ValueError("unsupported resident operand policy")
    if not isinstance(config.q_splits, int) or not 1 <= config.q_splits <= 4:
        raise ValueError("q_splits must be an integer in [1, 4]")
    if config.q_splits > (hq // hk) * n // config.block_m:
        raise ValueError("every Q split must contain a compute tile")
    if config.preprocess_block not in (64, 128, 256) or config.convert_block not in (1024, 2048, 4096):
        raise ValueError("unsupported preprocess/conversion block")
    if not isinstance(config.waves_per_eu, int) or not 0 <= config.waves_per_eu <= 16:
        raise ValueError("waves_per_eu must be an integer in [0, 16]")
    tensors = (q, k, v, o, do, lse)
    if (b < 1 or hq < 1 or hk < 1 or hq % hk or k.shape != (b, hk, n, d) or v.shape != k.shape or o.shape != q.shape
            or do.shape != q.shape or lse.shape != (b, hq, n)):
        raise ValueError("inconsistent attention tensor shapes")
    if q.device.type != "cuda" or any(not t.is_contiguous() or t.device != q.device for t in tensors):
        raise ValueError("inputs must be contiguous on the same GPU")
    if any(t.dtype != torch.bfloat16 for t in tensors[:-1]) or lse.dtype != torch.float32:
        raise ValueError("inputs must be BF16, with FP32 LSE")
    if triton.runtime.driver.active.get_current_target().arch != "gfx1250":
        raise ValueError("requires gfx1250")
    if (hq // hk) * n * 128 >= (1 << 29):
        raise ValueError("one query-head group's FP32 dQ workspace must fit in the signed 2-GiB buffer range")
    workspace = workspace if workspace is not None else allocate_workspace(q, k, config=config)
    if len(workspace) != (7 if config.q_splits > 1 else 5):
        raise ValueError("workspace does not match q_splits")
    delta, dqa, dq, dk, dv = workspace[:5]
    dka, dva = workspace[5:] if config.q_splits > 1 else (None, None)
    specifications = [(delta, q.shape[:-1], torch.float32),
                      (dqa, q.shape, torch.bfloat16 if config.packed_dq else torch.float32),
                      (dq, q.shape, torch.bfloat16), (dk, k.shape, torch.bfloat16), (dv, k.shape, torch.bfloat16)]
    if config.q_splits > 1:
        specifications += [(dka, k.shape, torch.float32), (dva, k.shape, torch.float32)]
    for value, shape, dtype in specifications:
        if value.shape != shape or value.dtype != dtype or value.device != q.device or not value.is_contiguous():
            raise ValueError("invalid backward workspace")
    _run_bwd_preprocess(o, do, delta, dqa, dk_acc=dka, dv_acc=dva, block_m=config.preprocess_block)
    _delayed_bwd_kernel[(hk, n // config.block_n, b * config.q_splits)](
        q, k, v, do, lse, delta, dqa, dka if dka is not None else dk, dva if dva is not None else dv, HQ=hq, HK=hk, N=n,
        SM_SCALE=d**-0.5 if sm_scale is None else sm_scale, BM=config.block_m, BN=config.block_n, NW=config.num_warps,
        TDM=config.use_tdm, STAGES=config.stages, DELAYED=config.delayed, UNROLL=config.unroll,
        FRAGMENT_FENCES=config.fragment_fences, FRAGMENT_FENCE_MASK=config.fragment_fence_mask,
        FRAGMENTED=config.fragmented, RESIDENT_KV=config.resident_kv, ORDERED=config.ordered,
        PACKED_DQ=config.packed_dq, REMATERIALIZE=config.rematerialize, OUTER_M=config.outer_m,
        ASYNC_STATS=config.async_stats, QUEUE_WMMA=config.queue_wmma, Q_SPLITS=config.q_splits, REUSE_Q=config.reuse_q,
        PINGPONG=config.pingpong, SPREAD_DQ=config.spread_dq, **_compiler_options(config))
    _convert_outputs[(triton.cdiv(q.numel(),
                                  config.convert_block), )](dqa, dq, dka if dka is not None else dqa,
                                                            dva if dva is not None else dqa, dk, dv, q.numel(),
                                                            k.numel(), d**-0.5 if sm_scale is None else sm_scale,
                                                            config.convert_block, PACKED=config.packed_dq,
                                                            PARTIALS=config.q_splits > 1, num_warps=4)
    return dq, dk, dv
