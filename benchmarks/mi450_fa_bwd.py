"""Correctness, tuning, and matched aiter comparison for gfx1250 backward.

Run GPU modes under gpu-lock. ``compile`` uses an explicit target and no GPU.
Results are appended as JSONL; ``--artifacts`` saves the generated assembly.
"""
import argparse
from collections import Counter, defaultdict
import ctypes
import dataclasses
import itertools
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.tlx.ops.kernels.flash_attn.gfx1250_bwd import (
    _compiler_options,
    _delayed_bwd_kernel,
    allocate_workspace,
    fa_backward,
    select_config,
)


@triton.jit
def _forward_state(Q, K, V, O, LSE, HQ: tl.constexpr, HK: tl.constexpr, N: tl.constexpr):
    head = tl.program_id(1)
    kv_head = head // HQ * HK + head % HQ // (HQ // HK)
    rows = tl.program_id(0) * 64 + tl.arange(0, 64)
    cols = tl.arange(0, 128)
    q = tl.load(Q + head.to(tl.int64) * N * 128 + rows[:, None] * 128 + cols[None, :])
    acc = tl.zeros((64, 128), tl.float32)
    maximum = tl.full((64, ), float("-inf"), tl.float32)
    total = tl.zeros((64, ), tl.float32)
    for start in range(0, N, 64):
        keys = start + tl.arange(0, 64)
        offsets = kv_head.to(tl.int64) * N * 128 + keys[:, None] * 128 + cols[None, :]
        k = tl.load(K + offsets)
        v = tl.load(V + offsets)
        scores = tl.dot(q, tl.trans(k)) * (128**-0.5 * 1.4426950408889634)
        new_max = tl.maximum(maximum, tl.max(scores, 1))
        alpha = tl.exp2(maximum - new_max)
        p = tl.exp2(scores - new_max[:, None])
        total = total * alpha + tl.sum(p, 1)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * alpha[:, None])
        maximum = new_max
    tl.store(O + head.to(tl.int64) * N * 128 + rows[:, None] * 128 + cols[None, :], acc / total[:, None])
    tl.store(LSE + head.to(tl.int64) * N + rows, (maximum + tl.log2(total)) * 0.6931471805599453)


def make_case(shape, reference=False, seed=0):
    b, hq, hk, n = shape
    torch.manual_seed(seed)
    q, do = [torch.randn((b, hq, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    k, v = [torch.randn((b, hk, n, 128), device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    o = torch.empty_like(q)
    lse = torch.empty((b, hq, n), device="cuda", dtype=torch.float32)
    if not reference:
        _forward_state[(n // 64, b * hq)](q, k, v, o, lse, hq, hk, n, num_warps=4)
        return (q, k, v, o, do, lse), None
    dq = torch.empty_like(q, dtype=torch.float32)
    dk, dv = [torch.zeros_like(k, dtype=torch.float32) for _ in range(2)]
    for batch in range(b):
        for head in range(hq):
            kh = head // (hq // hk)
            qr = q[batch, head].float().requires_grad_()
            kr = k[batch, kh].float().requires_grad_()
            vr = v[batch, kh].float().requires_grad_()
            scores = qr @ kr.T * 128**-0.5
            probs = scores.softmax(-1)
            out = probs @ vr
            grads = torch.autograd.grad(out, (qr, kr, vr), do[batch, head].float())
            o[batch, head] = out.detach()
            lse[batch, head] = scores.detach().logsumexp(-1)
            dq[batch, head] = grads[0]
            dk[batch, kh] += grads[1]
            dv[batch, kh] += grads[2]
    return (q, k, v, o, do, lse), (dq, dk, dv)


def errors(actual, expected):
    result = {}
    for name, a, e in zip(("dq", "dk", "dv"), actual, expected):
        diff = a.float() - e.float()
        nrmse = (diff.square().mean() / e.float().square().mean()).sqrt().item()
        max_abs = diff.abs().max().item()
        result[name] = {"nrmse": nrmse, "max_abs": max_abs}
        if not torch.isfinite(a).all().item() or not nrmse < 0.012:
            raise AssertionError(f"{name}: {result[name]}")
    return result


class AiterBackward:
    """Call the default gfx1250 ASM kernels, including fresh dQ zeroing."""

    def __init__(self, case, library, split_batches=True):
        self.case = case
        q, k, v, o, do, lse = case
        self.b, self.hq, self.n, _ = q.shape
        self.hk = k.shape[1]
        # The default gfx1250 binary miscomputes dQ beyond 4 GiB across
        # batches. Rebase independent batch chunks; retain an unsplit mode
        # for reproducing the failure under the correctness guard.
        per_batch_bytes = self.hq * self.n * 128 * 4
        max_batch = max(1, (1 << 32) // per_batch_bytes) if split_batches else self.b
        self.batch_chunks = [min(max_batch, self.b - start) for start in range(0, self.b, max_batch)]
        self.delta = torch.empty_like(lse)
        self.dqa = torch.empty_like(q, dtype=torch.float32)
        self.dq = torch.empty_like(q)
        self.dk_exp = torch.empty_like(q)
        self.dv_exp = torch.empty_like(q)
        self.dk, self.dv = torch.empty_like(k), torch.empty_like(v)
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.fn = self.lib.mi450_aiter_bwd
        self.fn.argtypes = [ctypes.c_void_p] * 11 + [ctypes.c_int] * 4 + [ctypes.c_float, ctypes.c_void_p]
        self.fn.restype = ctypes.c_float

    def __call__(self):
        q, k, v, o, do, lse = self.case
        tensors = (q, k, v, o, do, lse, self.delta, self.dqa, self.dq, self.dk_exp, self.dv_exp)
        start = 0
        for chunk in self.batch_chunks:
            status = self.fn(*(t[start:].data_ptr() for t in tensors), chunk, self.hq, self.hk, self.n, 128**-0.5,
                             torch.cuda.current_stream().cuda_stream)
            if status < 0:
                raise RuntimeError(f"aiter did not select a kernel: {status}")
            start += chunk
        if self.hq != self.hk:
            shape = (self.b, self.hk, self.hq // self.hk, self.n, 128)
            torch.sum(self.dk_exp.view(shape), dim=2, out=self.dk)
            torch.sum(self.dv_exp.view(shape), dim=2, out=self.dv)
        else:
            self.dk, self.dv = self.dk_exp, self.dv_exp
        return self.dq, self.dk, self.dv


def inspect_assembly(assembly):
    """Count the steady loop using backward branches, retaining source attribution.

    Counts are static ISA instructions per loop body and per wave, not profiler
    cycles. Select the WMMA-containing backward region with the most WMMAs;
    when several branches close it, retain the complete region.
    """
    lines = assembly.splitlines()
    labels = {}
    branches = []
    for index, line in enumerate(lines):
        label = re.match(r"^([.\w]+):", line)
        if label:
            labels[label[1]] = index
        branch = re.match(r"\s*s_(?:cbranch_\w+|branch)\s+([.\w]+)", line)
        if branch and branch[1] in labels and labels[branch[1]] < index:
            start = labels[branch[1]]
            count = sum("v_wmma_" in item for item in lines[start:index + 1])
            if count:
                branches.append((count, index - start, start, index + 1))
    if not branches:
        return {"loop_found": False}
    _, _, start, end = max(branches)
    counts = Counter()
    lds_sources = defaultdict(Counter)
    atomic_addresses = defaultdict(set)
    atomic_offsets = defaultdict(list)
    atomic_wmma_positions = defaultdict(list)
    wmma_position = 0
    atomic_clauses = []
    clause_remaining = 0
    # Inherit the debug location immediately preceding the loop.
    location = "unknown"
    for index, line in enumerate(lines[:end]):
        if line.lstrip().startswith(".loc"):
            locations = re.findall(r"([\w.]+\.py):(\d+):(\d+)", line)
            if locations:
                filename, row, _ = locations[-1]
                location = f"{filename}:{row}"
        if index < start:
            continue
        instruction = re.match(r"\s+([a-z][\w.]*)\s", line)
        if not instruction:
            continue
        opcode = instruction[1]
        counts[opcode] += 1
        if opcode.startswith("v_wmma_"):
            wmma_position += 1
        if opcode == "s_clause":
            clause_remaining = int(line.split()[1], 0) + 1
            atomic_clauses.append(0)
        elif clause_remaining:
            clause_remaining -= 1
            if opcode.startswith("buffer_atomic_"):
                atomic_clauses[-1] += 1
        if opcode.startswith("buffer_atomic_"):
            atomic_wmma_positions[opcode].append(wmma_position)
            address = re.search(r",\s*(v\d+)(?:\s*/\*(v\d+)\*/)?\s*,\s*s\[", line)
            if address:
                atomic_addresses[opcode].add(address[2] or address[1])
            immediate = re.search(r"\boffset:(\d+)", line)
            atomic_offsets[opcode].append(int(immediate[1]) if immediate else 0)
        if opcode.startswith("ds_load"):
            lds_sources[location][opcode] += 1
    return {
        "loop_found": True,
        "loop_start_line": start + 1,
        "loop_end_line": end,
        "loop_instruction_counts": dict(sorted(counts.items())),
        "loop_atomic_address_registers": {key: sorted(value)
                                          for key, value in sorted(atomic_addresses.items())},
        "loop_atomic_immediate_offsets_bytes": dict(sorted(atomic_offsets.items())),
        "loop_atomic_wmma_positions": dict(sorted(atomic_wmma_positions.items())),
        "loop_atomic_clause_sizes": [size for size in atomic_clauses if size],
        "loop_lds_loads": sum(value for key, value in counts.items() if key.startswith("ds_load")),
        "loop_lds_loads_by_source": {key: dict(value)
                                     for key, value in sorted(lds_sources.items())},
        "loop_assembly": "\n".join(lines[start:end]) + "\n",
    }


def compile_kernel(config, artifacts, shape, *, compile_options=None):
    constants = dict(HQ=shape[1], HK=shape[2], N=shape[3], SM_SCALE=128**-0.5, BM=config.block_m, BN=config.block_n,
                     NW=config.num_warps, TDM=config.use_tdm, STAGES=config.stages, DELAYED=config.delayed,
                     UNROLL=config.unroll, FRAGMENT_FENCES=config.fragment_fences,
                     FRAGMENT_FENCE_MASK=config.fragment_fence_mask, FRAGMENTED=config.fragmented,
                     RESIDENT_KV=config.resident_kv, ORDERED=config.ordered, PACKED_DQ=config.packed_dq,
                     REMATERIALIZE=config.rematerialize, OUTER_M=config.outer_m, ASYNC_STATS=config.async_stats,
                     QUEUE_WMMA=config.queue_wmma, Q_SPLITS=config.q_splits, REUSE_Q=config.reuse_q,
                     PINGPONG=config.pingpong, SPREAD_DQ=config.spread_dq)
    signature = {name: "*bf16" for name in ("Q", "K", "V", "DO", "DK", "DV")}
    signature.update({name: "*fp32" for name in ("LSE", "Delta")})
    signature["DQA"] = "*bf16" if config.packed_dq else "*fp32"
    if config.q_splits > 1:
        signature.update(DK="*fp32", DV="*fp32")
    # Match the pointer specialization of the fresh Torch allocations used
    # by make_case/allocate_workspace, including vector buffer-copy controls.
    attrs = {(_delayed_bwd_kernel.arg_names.index(name), ): [("tt.divisibility", 16)] for name in signature}
    kernel = triton.compile(ASTSource(_delayed_bwd_kernel, signature, constants,
                                      attrs=attrs), target=GPUTarget("hip", "gfx1250", 32),
                            options={**_compiler_options(config), **(compile_options or {})})
    if artifacts:
        artifacts.mkdir(parents=True, exist_ok=True)
        config_hash = hashlib.sha256(json.dumps(dataclasses.asdict(config), sort_keys=True).encode()).hexdigest()[:16]
        tag = "_".join(map(str, shape)) + "_" + config_hash
        (artifacts / f"{tag}.config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2) + "\n")
        for extension in ("ttgir", "llir", "amdgcn"):
            (artifacts / f"{tag}.{extension}").write_text(kernel.asm[extension])
    assembly = kernel.asm["amdgcn"]
    metadata = {
        "shared": kernel.metadata.shared,
        "wmma": assembly.count("v_wmma_"),
        "tdm": assembly.count("tensor_load_to_lds"),
        "pointer_alignment": 16,
    }
    for field in ("vgpr_count", "sgpr_count", "private_segment_fixed_size"):
        metadata[field] = int(re.search(r"\." + field + r":\s*(\d+)", assembly).group(1))
    analysis = inspect_assembly(assembly)
    loop_assembly = analysis.pop("loop_assembly", "")
    if artifacts and loop_assembly:
        (artifacts / f"{tag}.loop.s").write_text(loop_assembly)
    metadata.update(analysis)
    if compile_options:
        metadata["compile_options"] = compile_options
    return metadata


def benchmark(fn, repeats):
    import triton.testing
    return [triton.testing.do_bench_cudagraph(fn, rep=repeats, return_mode="median") for _ in range(3)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("compile", "check", "tune", "pipeline", "bench"), default="check")
    parser.add_argument("--shape", default="1,4,2,512", help="B,Hq,Hkv,N")
    parser.add_argument("--config", help="JSON BackwardConfig fields")
    parser.add_argument("--output", type=Path, default=Path("/host-tmp/mi450-bwd-results.jsonl"))
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--aiter-library", type=Path)
    parser.add_argument("--aiter-unsplit", action="store_true", help="Reproduce the raw aiter batch-addressing failure")
    parser.add_argument("--repeat-ms", type=int, default=200)
    parser.add_argument("--num-cus", type=int, default=192, help="CU count for offline compile policy selection")
    args = parser.parse_args()
    shape = tuple(map(int, args.shape.split(",")))
    overrides = json.loads(args.config) if args.config else {}
    config = select_config(shape[-1], *shape[:3], num_cus=args.num_cus if args.mode == "compile" else None,
                           packed_dq=overrides.get("packed_dq", True))
    config = dataclasses.replace(config, **overrides)
    kernel_sha256 = hashlib.sha256(
        Path(__file__).parents[1].joinpath(
            "third_party/tlx/ops/kernels/flash_attn/gfx1250_bwd.py").read_bytes()).hexdigest()

    def emit(record):
        record["time"] = time.time()
        record["kernel_sha256"] = kernel_sha256
        record["experiment"] = os.environ.get("MI450_EXPERIMENT", "initial")
        print(json.dumps(record), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a") as f:
            f.write(json.dumps(record) + "\n")

    if args.mode == "compile":
        emit({
            "mode": "compile", "shape": shape, "config": dataclasses.asdict(config),
            **compile_kernel(config, args.artifacts, shape)
        })
        return
    if not os.environ.get("GPU_LOCK_HELD"):
        raise RuntimeError("Run GPU modes under gpu-lock and pass GPU_LOCK_HELD through Docker")
    case, reference = make_case(shape, reference=args.mode == "check")
    workspace = allocate_workspace(case[0], case[1], config=config)
    configs = [config]
    if args.mode in ("tune", "pipeline"):
        if args.mode == "tune":
            configs = [
                dataclasses.replace(config, block_m=bm, block_n=bn, num_warps=nw, use_tdm=tdm,
                                    async_stats=config.async_stats and tdm)
                for bm, bn, nw, tdm in itertools.product((32, 64), (128, 256), (2, 4, 8), (False, True))
            ]
        else:
            configs = [
                dataclasses.replace(config, stages=stages, unroll=unroll)
                for stages, unroll in itertools.product((2, 3, 4), (1, 2, 4, 8))
            ]
        check_case, check_reference = make_case((1, 4, 2, 512), reference=True)
        check_workspace = allocate_workspace(check_case[0], check_case[1], config=config)
    for config in configs:
        record = {"mode": args.mode, "shape": shape, "provider": "tlx", "config": dataclasses.asdict(config)}
        try:
            if args.mode == "check":
                record["errors"] = errors(fa_backward(*case, config=config, workspace=workspace), reference)
                # A second invocation catches missing per-call accumulator clearing.
                record["repeat_errors"] = errors(fa_backward(*case, config=config, workspace=workspace), reference)
            else:
                if args.mode in ("tune", "pipeline"):
                    record["errors"] = errors(fa_backward(*check_case, config=config, workspace=check_workspace),
                                              check_reference)
                samples = benchmark(lambda: fa_backward(*case, config=config, workspace=workspace), args.repeat_ms)
                record.update(ms=statistics.median(samples), samples_ms=samples,
                              tflops=10 * shape[0] * shape[1] * shape[3]**2 * 128 / statistics.median(samples) / 1e9)
            emit(record)
        except Exception as exc:
            if args.mode not in ("tune", "pipeline"):
                raise
            emit({**record, "error": str(exc)})
    if args.aiter_library:
        aiter = AiterBackward(case, args.aiter_library, split_batches=not args.aiter_unsplit)
        record = {"mode": args.mode, "shape": shape, "provider": "aiter", "batch_chunks": aiter.batch_chunks}
        if args.mode == "check":
            record["errors"] = errors(aiter(), reference)
            record["repeat_errors"] = errors(aiter(), reference)
        else:
            if args.mode == "bench":
                record["tlx_vs_aiter_errors"] = errors(tuple(workspace[2:5]), aiter())
            samples = benchmark(aiter, args.repeat_ms)
            record.update(ms=statistics.median(samples), samples_ms=samples,
                          tflops=10 * shape[0] * shape[1] * shape[3]**2 * 128 / statistics.median(samples) / 1e9)
        emit(record)


if __name__ == "__main__":
    main()
