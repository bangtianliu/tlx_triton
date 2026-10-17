"""Public ``tlx.ops.flash_attn`` coverage for gfx950."""

import subprocess
import sys
from unittest import mock

import pytest
import torch

from triton._internal_testing import is_hip_cdna4


def _gfx950_device_indices():
    return [
        index for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "").startswith("gfx950")
    ]


def _has_two_gfx950_devices():
    return len(_gfx950_device_indices()) >= 2


def test_gfx950_device_indices_ignore_non_rocm_and_other_amd_devices():
    properties = (
        mock.Mock(spec=["name"]),
        mock.Mock(gcnArchName="gfx950:sramecc+:xnack-"),
        mock.Mock(gcnArchName="gfx942:sramecc+:xnack-"),
        mock.Mock(gcnArchName="gfx950"),
    )
    with (
            mock.patch.object(torch.cuda, "device_count", return_value=len(properties)),
            mock.patch.object(torch.cuda, "get_device_properties", side_effect=lambda index: properties[index]),
    ):
        assert _gfx950_device_indices() == [1, 3]
        assert _has_two_gfx950_devices()


def test_flash_attn_gfx950_catalog_entry():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    specs = [spec for spec in CATALOG if (spec.op, spec.arch) == ("flash_attn", "gfx950")]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.impl == "triton.tlx.ops.kernels.flash_attn.gfx950:flash_attn"
    assert spec.supports_backward

    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=64)
    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=128)
    with pytest.raises(InvalidInput, match="does not support float16"):
        check_inputs(spec, dtype=torch.float16, HEAD_DIM=64)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=256)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_catalog_resolves_public_implementation():
    from triton.tlx.ops._catalog import impl_for
    from triton.tlx.ops.kernels.flash_attn.gfx950 import flash_attn

    implementation, spec = impl_for("flash_attn", arch="gfx950")
    assert implementation is flash_attn
    assert spec.supports_backward


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize(
    ("requires_grad", "grad_enabled"),
    ((False, True), (True, False)),
)
def test_flash_attn_gfx950_inference_skips_backward_state(requires_grad, grad_enabled):
    from triton.tlx.ops.kernels.flash_attn import gfx950

    q, k, v = _qkv(64, requires_grad=requires_grad, context=256)
    sentinel = object()
    with (
            mock.patch.object(gfx950, "_flash_attn_inference", return_value=sentinel) as inference,
            mock.patch.object(gfx950._attention, "apply") as differentiable,
            torch.set_grad_enabled(grad_enabled),
    ):
        assert gfx950.flash_attn(q, k, v, space="smoke") is sentinel

    inference.assert_called_once_with(q, k, v, 64**-0.5, False, "smoke")
    differentiable.assert_not_called()


@pytest.mark.skipif(not _has_two_gfx950_devices(), reason="requires two gfx950 GPUs")
def test_flash_attn_gfx950_uses_input_device_when_current_device_differs():
    previous_device, query_device = _gfx950_device_indices()[:2]
    # Keep an illegal cross-device launch from poisoning the pytest process if
    # this regresses. The child deliberately leaves one gfx950 current while
    # the public op's inputs live on another gfx950.
    script = r"""
import sys

import torch

from triton.tlx.ops import flash_attn

previous_device = int(sys.argv[1])
query_device = int(sys.argv[2])
torch.cuda.set_device(previous_device)
q, k, v = [
    torch.randn(
        (1, 1, 256, 64),
        device=f"cuda:{query_device}",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    for _ in range(3)
]
assert torch.cuda.current_device() == previous_device

out = flash_attn(q, k, v, space="smoke")
assert out.device == q.device
assert torch.cuda.current_device() == previous_device
torch.cuda.synchronize(query_device)
torch.cuda.synchronize(previous_device)

grad_out = torch.randn_like(out)
out.backward(grad_out)
torch.cuda.synchronize(query_device)
torch.cuda.synchronize(previous_device)
assert torch.cuda.current_device() == previous_device
assert q.grad.device == k.grad.device == v.grad.device == q.device

rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
with torch.cuda.device(query_device):
    reference = torch.nn.functional.scaled_dot_product_attention(rq, rk, rv)
    reference.backward(grad_out)
torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)
for actual, expected in zip((q.grad, k.grad, v.grad), (rq.grad, rk.grad, rv.grad), strict=True):
    relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(expected.float())
    assert relative_l2.item() < 5e-3
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(previous_device),
         str(query_device)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"


def _qkv(head_dim, *, requires_grad, context=None):
    context = context or (512 if head_dim == 64 else 1024)
    return [
        torch.randn(
            (1, 4, context, head_dim),
            device="cuda",
            dtype=torch.bfloat16,
        ).requires_grad_(requires_grad) for _ in range(3)
    ]


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_gfx950_forward_returns_natural_log_lse(causal):
    from triton.tlx.ops.kernels.flash_attn.gfx950 import _flash_attn_forward

    torch.manual_seed(1)
    q, k, v = _qkv(64, requires_grad=False, context=256)
    scale = q.shape[-1]**-0.5

    _out, lse = _flash_attn_forward(q, k, v, scale, causal, "smoke")
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.ones((q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool).triu(1)
        scores.masked_fill_(causal_mask, float("-inf"))
    expected = torch.logsumexp(scores, dim=-1)

    assert lse.dtype is torch.float32
    assert lse.shape == q.shape[:-1]
    torch.testing.assert_close(lse, expected, atol=2e-2, rtol=2e-3)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("head_dim", (64, 128))
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_gfx950_forward_and_backward(head_dim, causal):
    from triton.tlx.ops import flash_attn

    torch.manual_seed(0)
    q, k, v = _qkv(head_dim, requires_grad=True)
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    if head_dim == 64 and not causal:
        grad_out = torch.randn((*q.shape[:-1], 2 * head_dim), device=q.device, dtype=q.dtype)[..., ::2]
        assert not grad_out.is_contiguous()
    else:
        grad_out = torch.randn_like(q)
    scale = None if causal else 0.75 * head_dim**-0.5

    if head_dim == 64 and not causal:
        # Exercise the public default and the STORE_LSE/ALLOW_PRESCALE_Q
        # autotuner-key ABI; the remaining cases use the smaller smoke space.
        out = flash_attn(q, k, v, causal=causal, sm_scale=scale)
    else:
        out = flash_attn(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(
        rq,
        rk,
        rv,
        is_causal=causal,
        scale=scale,
    )

    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)
    out.backward(grad_out)
    reference.backward(grad_out)

    for name, actual, expected in zip(
        ("dq", "dk", "dv"),
        (q.grad, k.grad, v.grad),
        (rq.grad, rk.grad, rv.grad),
            strict=True,
    ):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        tolerance = 5e-3 if head_dim == 64 else 1e-2
        assert relative_l2.item() < tolerance, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_honors_deterministic_algorithms():
    from triton.tlx.ops import flash_attn
    from triton.tlx.ops.kernels.flash_attn import gfx950, gfx950_bwd

    q, k, v = _qkv(128, requires_grad=True, context=256)
    forward_state = (
        torch.empty_like(q),
        torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32),
    )
    backward_result = tuple(torch.empty_like(tensor) for tensor in (q, k, v))
    grad_out = torch.empty_like(q)
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        with (
                mock.patch.object(gfx950, "_flash_attn_forward", return_value=forward_state),
                mock.patch.object(gfx950_bwd, "fa_backward", return_value=backward_result) as backward,
        ):
            torch.use_deterministic_algorithms(False)
            with mock.patch.object(gfx950_bwd, "fa_backward_is_deterministic") as classify_determinism:
                out = flash_attn(q, k, v, space="smoke")
                out.backward(grad_out)
            classify_determinism.assert_not_called()
            backward.assert_called_once()

            backward.reset_mock()
            torch.use_deterministic_algorithms(True)
            out = flash_attn(q, k, v, space="smoke")
            with pytest.raises(RuntimeError, match="does not have a deterministic implementation"):
                out.backward(grad_out)
            backward.assert_not_called()

            torch.use_deterministic_algorithms(True, warn_only=True)
            out = flash_attn(q, k, v, space="smoke")
            with pytest.warns(UserWarning, match="does not have a deterministic implementation"):
                out.backward(grad_out)
            backward.assert_called_once()

            backward.reset_mock()
            torch.use_deterministic_algorithms(True)
            with mock.patch.object(gfx950_bwd, "fa_backward_is_deterministic", return_value=True):
                out = flash_attn(q, k, v, space="smoke")
                out.backward(grad_out)
            backward.assert_called_once()
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "family", "expected"),
    (
        ((1, 4, 256, 64), (1, 4, 256, 64), "noncausal_direct_n256", True),
        ((1, 4, 4096, 64), (1, 4, 4096, 64), "noncausal_fused_n256", False),
        ((16, 27, 256, 128), (16, 27, 256, 128), None, False),
        ((16, 27, 200, 128), (16, 27, 200, 128), None, True),
        ((32, 1, 2600, 256), (32, 1, 2600, 256), None, True),
    ),
)
def test_flash_attn_gfx950_backward_determinism_matches_route(q_shape, k_shape, family, expected):
    from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

    q = mock.Mock(shape=q_shape)
    k = mock.Mock(shape=k_shape)
    tensors = (q, k, mock.Mock(), mock.Mock(), mock.Mock(), mock.Mock())
    dispatch = mock.Mock(family=family)
    with (
            mock.patch.object(gfx950_bwd, "_validate_inputs"),
            mock.patch.object(gfx950_bwd, "_select_d64_dispatch_for_device", return_value=dispatch) as select_d64,
    ):
        actual = gfx950_bwd.fa_backward_is_deterministic(*tensors, 0.125, False)

    assert actual is expected
    assert select_d64.call_count == int(q_shape[-1] == 64)


@pytest.mark.parametrize(
    ("shape", "causal", "split_q", "planes"),
    (
        pytest.param((16, 64, 8, 1024, 128), False, 1, 1, id="pr0-gqa1k"),
        pytest.param((16, 64, 8, 2048, 128), False, 1, 1, id="pr1-gqa2k"),
        pytest.param((16, 64, 8, 4096, 128), False, 1, 1, id="pr2-gqa4k"),
        pytest.param((16, 64, 8, 8192, 128), False, 1, 1, id="pr3-gqa8k"),
        pytest.param((15, 64, 8, 16384, 128), False, 1, 1, id="pr4-gqa16k"),
        pytest.param((16, 16, 16, 1024, 128), False, 1, 1, id="pr5-mha1k"),
        pytest.param((16, 16, 16, 2048, 128), False, 1, 1, id="pr6-mha2k"),
        pytest.param((16, 16, 16, 4096, 128), False, 1, 1, id="pr7-mha4k"),
        pytest.param((16, 16, 16, 8192, 128), False, 1, 1, id="pr8-mha8k"),
        pytest.param((16, 16, 16, 16384, 128), False, 1, 2, id="pr9-mha16k"),
        pytest.param((16, 16, 16, 16384, 128), False, None, 2, id="pr9-auto"),
        pytest.param((16, 16, 16, 16384, 128), True, None, 1, id="pr9-causal-auto"),
        pytest.param((16, 16, 16, 16384, 128), True, 1, 1, id="pr9-causal-unsplit"),
        pytest.param((16, 16, 16, 16384, 128), False, 2, 1, id="pr9-split2"),
        pytest.param((15, 16, 16, 16384, 128), False, 1, 1, id="batch15"),
        pytest.param((16, 32, 32, 16384, 128), False, 1, 1, id="heads32"),
        pytest.param((16, 16, 8, 16384, 128), False, 1, 1, id="gqa2"),
        pytest.param((16, 16, 16, 16128, 128), False, 1, 1, id="shorter"),
        pytest.param((16, 16, 16, 16640, 128), False, 1, 1, id="longer"),
    ),
)
def test_flash_attn_gfx950_d128_accumulator_allocation_metadata(shape, causal, split_q, planes):
    from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

    batch, hq, hkv, n_ctx, head_dim = shape
    q = torch.empty((batch, hq, n_ctx, head_dim), device="meta", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, n_ctx, head_dim), device="meta", dtype=torch.bfloat16)
    accumulator = gfx950_bwd._allocate_dq_accumulator(q, k, causal=causal, split_q=split_q)

    assert accumulator.shape == ((2, *q.shape) if planes == 2 else q.shape)
    assert accumulator.dtype is torch.bfloat16
    assert accumulator.device == q.device
    assert accumulator.is_contiguous()
    logical_bytes = accumulator.numel() * accumulator.element_size()
    assert logical_bytes == planes * q.numel() * q.element_size()
    assert accumulator.untyped_storage().nbytes() == logical_bytes
    if planes == 2:
        # Metadata tensors verify the 2 GiB contract without allocating GPU storage.
        assert logical_bytes == 2**31


@pytest.mark.parametrize(
    ("shape", "causal", "split_q", "planes"),
    (
        pytest.param((16, 16, 16, 16384, 128), False, None, 2, id="planes2-auto"),
        pytest.param((16, 16, 16, 16384, 128), False, 1, 2, id="planes2-unsplit"),
        pytest.param((16, 16, 16, 16384, 128), True, 1, 1, id="causal"),
        pytest.param((16, 16, 16, 16384, 128), False, 2, 1, id="split2"),
        pytest.param((16, 16, 16, 8192, 128), False, 1, 1, id="mha8k"),
        pytest.param((16, 64, 8, 8192, 128), False, 1, 1, id="long-gqa"),
    ),
)
@pytest.mark.parametrize("invalid", ("shape", "dtype", "device", "noncontiguous-storage"))
def test_flash_attn_gfx950_d128_accumulator_rejects_invalid_metadata(shape, causal, split_q, planes, invalid):
    from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

    batch, hq, hkv, n_ctx, head_dim = shape
    q = torch.empty((batch, hq, n_ctx, head_dim), device="meta", dtype=torch.bfloat16)
    k = torch.empty((batch, hkv, n_ctx, head_dim), device="meta", dtype=torch.bfloat16)
    accumulator_shape = (2, *q.shape) if planes == 2 else tuple(q.shape)
    if invalid == "shape":
        wrong_shape = tuple(q.shape) if planes == 2 else (2, *q.shape)
        accumulator = torch.empty(wrong_shape, device=q.device, dtype=q.dtype)
    elif invalid == "dtype":
        accumulator = torch.empty(accumulator_shape, device=q.device, dtype=torch.float32)
    elif invalid == "device":
        # Replace only device metadata to avoid a 2 GiB CPU allocation.
        accumulator = mock.Mock(shape=accumulator_shape, dtype=q.dtype, device=torch.device("cpu"))
        accumulator.is_contiguous.return_value = True
    else:
        padded = torch.empty((*accumulator_shape[:-1], 2 * head_dim), device=q.device, dtype=q.dtype)
        accumulator = padded[..., ::2]
        assert accumulator.shape == accumulator_shape
        assert not accumulator.is_contiguous()

    class LaunchTrap:
        def __getitem__(self, grid):
            pytest.fail("Invalid accumulator reached a GPU launch")

    with (
            mock.patch.object(gfx950_bwd, "_attn_bwd_dkdv_dq_d128_gqa_kernel", LaunchTrap()),
            mock.patch.object(gfx950_bwd, "_attn_bwd_dkdv_dq_d128_gqa_bm32_kernel", LaunchTrap()),
            pytest.raises(ValueError, match="dQ accumulator shape, dtype, device, or strides"),
    ):
        gfx950_bwd._run_bwd_d128_gqa(
            q, k, k, q, None, None, accumulator, q, k, k, head_dim**-0.5, causal, split_q=split_q,
        )


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize(("batch", "heads", "n_ctx"), ((1, 1, 128), (2, 3, 384)))
def test_flash_attn_gfx950_d128_planes2_converter_merges_full_output(batch, heads, n_ctx):
    from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

    head_dim, block_m = 128, 128
    shape = (batch, heads, n_ctx, head_dim)
    indices = torch.arange(batch * heads * n_ctx * head_dim, dtype=torch.int32).reshape(shape)
    # Different periods vary both planes across rows, features, heads, and tiles.
    plane0 = ((indices % 509 - 254).float() / 64).to(torch.bfloat16)
    plane1 = (((indices * 37) % 257 - 128).float() / 128).to(torch.bfloat16)
    # Include cancellation and both BF16 rounding directions at a midpoint.
    plane0.flatten()[:4] = torch.tensor((4, 1 + 2**-7, 1, 1 + 2**-7), dtype=torch.bfloat16)
    plane1.flatten()[:4] = torch.tensor((-4, -1, 2**-8, 2**-8), dtype=torch.bfloat16)
    fp32_sum = plane0.float() + plane1.float()
    expected = fp32_sum.to(torch.bfloat16)
    assert torch.count_nonzero(fp32_sum != expected.float()) > 0
    torch.testing.assert_close(
        expected.flatten()[:4], torch.tensor((0, 2**-7, 1, 1 + 2**-6), dtype=torch.bfloat16), rtol=0, atol=0,
    )

    def pack_native(logical):
        # Logical bit order is m3,m2,m1,m0,d6,d5,d4,d3,d2,d1,d0.
        bits = logical.reshape(-1, *([2] * 11))
        # Native memory bit order is d5,d4,d6,d1,d3,d2,m3,m2,m1,m0,d0.
        return bits.permute(0, 6, 7, 5, 10, 8, 9, 1, 2, 3, 4, 11).contiguous().reshape(shape)

    accumulator = torch.stack((pack_native(plane0), pack_native(plane1))).to("cuda")
    accumulator_before = accumulator.clone()
    output = torch.full(shape, float("nan"), device="cuda", dtype=torch.bfloat16)
    gfx950_bwd._attn_bwd_dq_native_planes2_convert_kernel[(n_ctx // block_m, batch * heads)](
        accumulator, output, N=n_ctx, D=head_dim, BLOCK_M=block_m, PLANE_ELEMENTS=output.numel(),
        num_warps=4, matrix_instr_nonkdim=16,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(accumulator, accumulator_before, rtol=0, atol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("exact", (False, True), ids=("split", "exact"))
def test_flash_attn_gfx950_short_d128_forward_and_backward(causal, exact, monkeypatch):
    from triton.tlx.ops import flash_attn
    from triton.tlx.ops.kernels.flash_attn import gfx950_bwd

    route_options = (
        gfx950_bwd._D128_EXACT_ENABLE_ENV,
        gfx950_bwd._D128_PERSISTENT_ENABLE_ENV,
        gfx950_bwd._D128_PERSISTENT_PIPE_ENABLE_ENV,
        gfx950_bwd._D128_SINK_INSTS_ENV,
        gfx950_bwd._D128_REGCLASS_PRIORITY_ENV,
        gfx950_bwd._D128_REVERSE_LOCAL_ENV,
    )
    for option in route_options:
        monkeypatch.setenv(option, "0")
    monkeypatch.setenv(gfx950_bwd._D128_EXACT_ENABLE_ENV, str(int(exact)))

    shape = (16, 27, 200, 128)
    generator = torch.Generator(device="cuda").manual_seed(3735 + 2 * int(exact) + int(causal))
    q, k, v = [
        torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator).requires_grad_() for _ in range(3)
    ]
    rq, rk, rv = (tensor.detach().clone().requires_grad_() for tensor in (q, k, v))
    grad_out = torch.randn(shape, device="cuda", dtype=torch.bfloat16, generator=generator)

    out = flash_attn(q, k, v, causal=causal, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(rq, rk, rv, is_causal=causal)
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)

    out.backward(grad_out)
    reference.backward(grad_out)
    for name, actual, expected in zip(
        ("dq", "dk", "dv"),
        (q.grad, k.grad, v.grad),
        (rq.grad, rk.grad, rv.grad),
            strict=True,
    ):
        assert torch.isfinite(actual).all(), name
        relative_l2 = torch.linalg.vector_norm(actual.float() - expected.float()) / torch.linalg.vector_norm(
            expected.float())
        assert relative_l2.item() < 1e-2, (name, relative_l2.item())


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_fp16():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = [torch.randn((1, 1, 128, 64), device="cuda", dtype=torch.float16) for _ in range(3)]
    with pytest.raises(InvalidInput, match="does not support float16"):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_unknown_space():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=False)
    with pytest.raises(InvalidInput, match="does not provide space"):
        flash_attn(q, k, v, space="heuristic")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("requires_grad", (False, True))
@pytest.mark.parametrize(
    ("invalid", "match"),
    (
        ("rank", "rank-4"),
        ("shape", "same shape"),
        ("dtype", "same dtype"),
    ),
)
def test_flash_attn_gfx950_invalid_inputs_use_public_exception(invalid, match, requires_grad):
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=requires_grad, context=256)
    if invalid == "rank":
        q = q[0]
        k = k[0]
        v = v[0]
    elif invalid == "shape":
        k = k[..., :-1, :].contiguous()
    else:
        k = k.to(torch.float16).requires_grad_(requires_grad)

    with pytest.raises(InvalidInput, match=match):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_keeps_inference_only_shapes():
    from triton.tlx.ops import flash_attn

    q, k, v = _qkv(64, requires_grad=False, context=192)
    out = flash_attn(q, k, v, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.testing.assert_close(out, reference, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
def test_flash_attn_gfx950_rejects_unsupported_backward_shape():
    from triton.tlx.ops import InvalidInput, flash_attn

    q, k, v = _qkv(64, requires_grad=True, context=192)
    with pytest.raises(InvalidInput, match="supported MHA shapes"):
        flash_attn(q, k, v, space="smoke")


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("operand", ("q", "k", "v"))
def test_flash_attn_gfx950_rejects_noncontiguous_backward_inputs(operand):
    from triton.tlx.ops import InvalidInput, flash_attn

    tensors = _qkv(64, requires_grad=True, context=256)
    padded = torch.randn(
        (*tensors[0].shape[:-1], 128),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    tensors[("q", "k", "v").index(operand)] = padded[..., ::2]
    assert not tensors[("q", "k", "v").index(operand)].is_contiguous()

    with pytest.raises(InvalidInput, match=rf"{operand} must be contiguous"):
        flash_attn(*tensors, space="smoke")


def _mxfp8_reference_quantize(x, *, sequence=False, square=False):
    # Independent mathematical RCEIL, without the implementation's bit tricks.
    x = x.float().transpose(-1, -2).contiguous() if sequence else x.float()
    rows, cols = x.shape[-2:]
    if square:
        blocks = x.reshape(*x.shape[:-2], rows // 32, 32, cols // 32, 32)
        maximum = blocks.abs().amax((-3, -1), keepdim=True)
    else:
        blocks = x.reshape(*x.shape[:-1], cols // 32, 32)
        maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum * (1.0 / 448.0))).clamp(-127, 127)
    scale = torch.exp2(exponent)
    payload = (blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    decoded = (payload.float() * scale).reshape(x.shape)
    if square:
        byte = (exponent.squeeze(-1).squeeze(-2) + 127).to(torch.uint8).repeat_interleave(32, -2)
    else:
        byte = (exponent.squeeze(-1) + 127).to(torch.uint8)
    payload = payload.reshape(x.shape)
    if sequence:
        payload = payload.transpose(-1, -2).contiguous()
        decoded = decoded.transpose(-1, -2).contiguous()
    return payload, byte, decoded


def test_flash_attn_mxfp8_gfx950_catalog_buffer_span():
    from triton.tlx.ops._catalog import CATALOG, InvalidInput, check_inputs

    spec, = [spec for spec in CATALOG if (spec.op, spec.arch) == ("flash_attn_mxfp8", "gfx950")]
    assert spec.supports_backward
    check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=128, N_CTX=2**24 - 256)
    with pytest.raises(InvalidInput, match="does not support these inputs"):
        check_inputs(spec, dtype=torch.bfloat16, HEAD_DIM=128, N_CTX=2**24)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("sequence", (False, True))
@pytest.mark.parametrize("strided", (False, True))
def test_flash_attn_mxfp8_gfx950_quantization_axes(sequence, strided):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_mxfp8

    torch.manual_seed(731)
    source = torch.randn((2, 3, 96, 256 if strided else 128), device="cuda", dtype=torch.bfloat16)
    x = source[..., ::2] if strided else source
    x[:, :, :32, :32] = 0
    x[:, :, 32:64] *= 2.0**-30
    x[:, :, 64:] *= 2.0**20
    actual, scales = quantize_mxfp8(x, transpose_for_reduction=sequence)
    expected, expected_scales, _ = _mxfp8_reference_quantize(x, sequence=sequence)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("sequence", (False, True))
def test_flash_attn_mxfp8_gfx950_quantization_scale_boundaries(sequence):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_mxfp8

    maxima = torch.tensor([448 * 2.0**-10, 448, 450, 448 * 2.0**10], device="cuda", dtype=torch.bfloat16)
    x = maxima.repeat_interleave(32).expand(1, 1, 128, 128).contiguous()
    if sequence:
        x = x.transpose(-1, -2).contiguous()
    actual, scales = quantize_mxfp8(x, transpose_for_reduction=sequence)
    expected, expected_scales, _ = _mxfp8_reference_quantize(x, sequence=sequence)
    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("sequence", (False, True))
@pytest.mark.parametrize("forward", (False, True))
def test_flash_attn_mxfp8_gfx950_quantization_minimum_scale(sequence, forward):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import quantize_mxfp8_head, quantize_mxfp8_v
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_mxfp8

    # The minimum E8M0 scale is subnormal FP32 (2**-127), not zero.
    # Include adjacent floats at its RCEIL transition, zero, and large values.
    maxima = torch.tensor([
        0, 1, 0x00800000, 0x04000000, 0x04600000, 0x04600001, 0x04600002, 0x04dfffff, 0x04e00000, 0x04e00001, 0x7f7fffff
    ], dtype=torch.int32).view(torch.float32)
    x = maxima[None, :, None, None].expand(1, len(maxima), 32, 128).contiguous()
    x[..., 1::2] *= -1
    # CPU double log/division avoids reference-side log rounding and FTZ.
    exponent = torch.ceil(torch.log2((maxima * (1.0 / 448.0)).double())).clamp(-127, 127)
    scales = torch.exp2(exponent)
    expected = (x.double() / scales[None, :, None, None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    if forward:
        actual, actual_scales = (quantize_mxfp8_v if sequence else quantize_mxfp8_head)(x.cuda())
    else:
        actual, actual_scales = quantize_mxfp8(x.cuda(), transpose_for_reduction=sequence)
    expected_scales = (exponent + 127).to(torch.uint8)[None, :, None, None].expand(actual_scales.shape)
    torch.testing.assert_close(actual_scales.cpu(), expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(actual.cpu().view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("n_ctx", (96, 256, 1024))
@pytest.mark.parametrize("transpose_storage", (False, True))
def test_flash_attn_mxfp8_gfx950_fused_quantization(n_ctx, transpose_storage):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_quant import quantize_backward_operands

    torch.manual_seed(631)
    tensors = [torch.randn((1, 2, n_ctx, 128), device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    tensors[0][:, :, :32, :32] = 0
    result = quantize_backward_operands(*tensors, transpose_storage=transpose_storage)
    for (actual, scales), (index, sequence) in zip(result, ((0, True), (1, True), (2, False), (3, False), (3, True))):
        expected, expected_scales, _ = _mxfp8_reference_quantize(tensors[index], sequence=sequence)
        torch.testing.assert_close(scales, expected_scales, atol=0, rtol=0)
        torch.testing.assert_close(actual.view(torch.uint8), expected.view(torch.uint8), atol=0, rtol=0)
        if transpose_storage and sequence:
            assert actual.stride(-2) == 1 and actual.stride(-1) == n_ctx


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("scale", (0.0, -0.5, 0.5, 1.3))
def test_flash_attn_mxfp8_gfx950_forward_returns_base2_lse(causal, scale):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import (_launch_quantized, quantize_mxfp8_head,
                                                                quantize_mxfp8_v)

    torch.manual_seed(20)
    q, k, v = [torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(3)]
    q8, qs = quantize_mxfp8_head(q)
    k8, ks = quantize_mxfp8_head(k)
    v8, vs = quantize_mxfp8_v(v)
    _, lse = _launch_quantized(q8, k8, v8, qs, ks, vs, causal, scale, return_lse=True)
    q_ref = q8.float() * torch.exp2(qs.float() - 127).repeat_interleave(32, -1)
    k_ref = k8.float() * torch.exp2(ks.float() - 127).repeat_interleave(32, -1)
    scores = (q_ref @ k_ref.transpose(-1, -2)) * scale
    if causal:
        row = torch.arange(256, device=q.device)
        scores.masked_fill_(row[:, None] < row[None, :], -float("inf"))
    expected = torch.logsumexp(scores, -1) * 1.4426950408889634
    torch.testing.assert_close(lse, expected, atol=3e-4, rtol=1e-5)


@pytest.mark.skipif(not _has_two_gfx950_devices(), reason="requires two gfx950 GPUs")
def test_flash_attn_mxfp8_gfx950_uses_input_device_when_current_device_differs():
    previous_device, query_device = _gfx950_device_indices()[:2]
    script = r"""
import sys
import torch
from triton.tlx.ops import flash_attn_mxfp8
previous, target = map(int, sys.argv[1:])
torch.cuda.set_device(previous)
q, k, v = [(torch.randn((1, 2, 256, 128), device=f"cuda:{target}", dtype=torch.bfloat16) * .5)
           .requires_grad_() for _ in range(3)]
out = flash_attn_mxfp8(q, k, v)
assert out.device == q.device and torch.cuda.current_device() == previous
out.backward(torch.randn_like(out))
assert torch.cuda.current_device() == previous
assert all(x.grad.device == q.device and bool(x.grad.isfinite().all()) for x in (q, k, v))
"""
    subprocess.run([sys.executable, "-c", script, str(previous_device), str(query_device)], check=True)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("n_ctx", (96, 256))
def test_flash_attn_mxfp8_gfx950_backward_matches_recipe(causal, n_ctx):
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950_bwd import launch_backward

    torch.manual_seed(20)
    q, k, v, do = [torch.randn((1, 2, n_ctx, 128), device="cuda", dtype=torch.bfloat16) * 0.5 for _ in range(4)]
    q8, qs, qf = _mxfp8_reference_quantize(q)
    k8, ks, kf = _mxfp8_reference_quantize(k)
    qdk, qdks, qdkf = _mxfp8_reference_quantize(q, sequence=True)
    kdq, kdqs, kdqf = _mxfp8_reference_quantize(k, sequence=True)
    vb, vs, vf = _mxfp8_reference_quantize(v)
    do8, dos, dof = _mxfp8_reference_quantize(do)
    dodv, dodvs, dodvf = _mxfp8_reference_quantize(do, sequence=True)
    sm_scale = 0.5
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        scores = (qf @ kf.transpose(-1, -2)) * sm_scale
        if causal:
            row = torch.arange(n_ctx, device=q.device)
            scores.masked_fill_(row[:, None] < row[None, :], -float("inf"))
        lse = torch.logsumexp(scores, -1) * 1.4426950408889634
        p = scores.softmax(-1)
        out = (p @ vf).to(torch.bfloat16)
        delta = (out.float() * do.float()).sum(-1)
        dp = dof @ vf.transpose(-1, -2)
        _, _, ds = _mxfp8_reference_quantize(p * (dp - delta[..., None]), square=True)
        p_quant = (p * 256).to(torch.float8_e4m3fn).float() / 256
        expected = (ds @ kdqf * sm_scale, ds.transpose(-1, -2) @ qdkf * sm_scale, p_quant.transpose(-1, -2) @ dodvf)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    dq = torch.empty_like(q, dtype=torch.float32)
    dk, dv = torch.empty_like(k), torch.empty_like(v)
    kernel = launch_backward(do8, dodv, q8, qdk, k8, kdq, vb, out, lse, qs, qdks, ks, kdqs, vs, dos, dodvs, sm_scale,
                             do, dq, dk, dv, torch.empty_like(lse), causal=causal, block_m=64, block_n=128)
    assert "v_mfma_scale_f32" in kernel.asm["amdgcn"]
    for actual, reference in zip((dq, dk, dv), expected):
        assert torch.isfinite(actual).all()
        # BF16 output rounding and near-tie FP8 dS rounding can change a few
        # elements; bound total error, not just gradient direction.
        relative_rms = ((actual.float() - reference).square().mean() / reference.square().mean()).sqrt()
        assert relative_rms < 0.01


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("n_ctx", (1024, 2048, 4096, 8192))
@pytest.mark.parametrize("scale", (128**-0.5, 0.5, 1.3))
@pytest.mark.parametrize("seed", (20, 21))
def test_flash_attn_mxfp8_gfx950_backward_meta_shapes(causal, n_ctx, scale, seed):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(seed)
    q, k, v = [(torch.randn((4, 32, n_ctx, 128), device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_()
               for _ in range(3)]
    rq, rk, rv = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    do = torch.randn_like(q)
    out = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale, space="smoke")
    reference = torch.nn.functional.scaled_dot_product_attention(rq, rk, rv, is_causal=causal, scale=scale)
    out.backward(do)
    reference.backward(do)
    if scale <= 0.5:
        torch.testing.assert_close(out, reference, atol=0.2, rtol=0)
    # At scale 1.3, Q/K FP8 quantization changes sharp softmax peaks: the
    # decoded-input FP32 reference also exceeds 0.2 at isolated elements.
    # Bound direction and magnitude against BF16 for every scale and tensor.
    for actual, expected in ((out, reference), (q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
        assert actual.dtype == torch.bfloat16
        assert torch.isfinite(actual).all()
        actual, expected = actual.float(), expected.float()
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0)
        relative_rms = ((actual - expected).square().mean() / expected.square().mean()).sqrt()
        assert cosine >= 0.98
        assert relative_rms < 0.15


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_gfx950_backward_zero_blocks(causal):
    from triton.tlx.ops import flash_attn_mxfp8

    q, k, v = [torch.zeros((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    out = flash_attn_mxfp8(q, k, v, causal=causal)
    out.backward(torch.zeros_like(out))
    for tensor in (out, q.grad, k.grad, v.grad):
        torch.testing.assert_close(tensor, torch.zeros_like(tensor), atol=0, rtol=0)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("causal", (False, True))
@pytest.mark.parametrize("scale", (0.0, -0.5))
@pytest.mark.parametrize("requires_grad", (False, True))
def test_flash_attn_mxfp8_gfx950_nonpositive_scale(causal, scale, requires_grad):
    from triton.tlx.ops import flash_attn_mxfp8

    torch.manual_seed(21)
    q, k, v = [(torch.randn((1, 2, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_(requires_grad)
               for _ in range(3)]
    rq, rk, rv = [x.detach().float().requires_grad_(requires_grad) for x in (q, k, v)]
    # Some SDPA implementations return NaN for negative scale; use explicit
    # FP32 softmax attention so the test does not bless a broken reference.
    scores = (rq @ rk.transpose(-1, -2)) * scale
    if causal:
        row = torch.arange(256, device=q.device)
        scores = scores.masked_fill(row[:, None] < row[None, :], -float("inf"))
    reference = scores.softmax(-1) @ rv
    actual = flash_attn_mxfp8(q, k, v, causal=causal, sm_scale=scale)
    torch.testing.assert_close(actual.float(), reference, atol=0.2, rtol=0)
    if requires_grad:
        do = torch.randn_like(actual)
        actual.backward(do)
        reference.backward(do.float())
        for got, ref in ((q.grad, rq.grad), (k.grad, rk.grad), (v.grad, rv.grad)):
            assert torch.isfinite(got).all()
            relative_rms = ((got.float() - ref).square().mean() / ref.square().mean().clamp_min(1e-30)).sqrt()
            assert relative_rms < 0.15


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("shape,scale,match", [
    ((0, 1, 256, 128), 0.5, "nonempty"),
    ((1, 0, 256, 128), 0.5, "nonempty"),
    ((1, 1, 0, 128), 0.5, "nonempty"),
    ((1, 1, 256, 128), float("nan"), "finite"),
    ((1, 1, 256, 128), float("inf"), "finite"),
    ((1, 1, 256, 128), -float("inf"), "finite"),
])
def test_flash_attn_mxfp8_gfx950_invalid_inputs(shape, scale, match):
    from triton.tlx.ops import InvalidInput, flash_attn_mxfp8

    q, k, v = [torch.empty(shape, device="cuda", dtype=torch.bfloat16) for _ in range(3)]
    with pytest.raises(InvalidInput, match=match):
        flash_attn_mxfp8(q, k, v, sm_scale=scale)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("warn_only", (False, True))
@pytest.mark.parametrize("causal", (False, True))
def test_flash_attn_mxfp8_gfx950_determinism(warn_only, causal):
    from triton.tlx.ops import flash_attn_mxfp8

    q, k, v = [(torch.randn((1, 1, 256, 128), device="cuda", dtype=torch.bfloat16) * 0.5).requires_grad_()
               for _ in range(3)]
    out = flash_attn_mxfp8(q, k, v, causal=causal)
    do = torch.randn_like(out)
    enabled = torch.are_deterministic_algorithms_enabled()
    warned = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
        expected = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
        for _ in range(3):
            actual = torch.autograd.grad(out, (q, k, v), do, retain_graph=True)
            for got, ref in zip(actual, expected):
                torch.testing.assert_close(got, ref, atol=0, rtol=0)
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warned)


@pytest.mark.skipif(not is_hip_cdna4(), reason="requires a gfx950 GPU")
@pytest.mark.parametrize("kind", ("head", "v", "v_transposed"))
def test_flash_attn_mxfp8_gfx950_quantization_large_head_offset_codegen(kind):
    import re
    from triton.tlx.ops.kernels.flash_attn_mxfp8.gfx950 import (_quantize_mxfp8_kernel, _quantize_mxfp8_v_kernel)

    x = torch.empty((1, 1, 64, 128), device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty((1, 1, 64, 4), device="cuda", dtype=torch.uint8)
    # Compile only: head 2's source offset would be 2**31 elements. Dummy
    # tensors stay tiny and the deliberately oversized strides are never run.
    args = (x, y, scales, 2**30, 2**30, 128, 1, 3, 64)
    if kind == "head":
        kernel = _quantize_mxfp8_kernel.warmup(*args, HEAD_DIM=128, BLOCK_N=64, PACK_K=False, num_warps=4, grid=(1, 3))
    else:
        kernel = _quantize_mxfp8_v_kernel.warmup(*args, HEAD_DIM=128, BLOCK_N=64, TRANSPOSED=kind == "v_transposed",
                                                 num_warps=1, grid=(1, 3))
    ir = kernel.asm["ttir"]
    pid = re.search(r"(%[\w]+) = tt.get_program_id y : i32", ir).group(1)
    widened = re.search(rf"(%[\w]+) = arith.extsi {re.escape(pid)} : i32 to i64", ir)
    assert widened is not None, "widen head index before multiplying tensor strides"
    assert re.search(r"arith.muli .* : i64", ir)
