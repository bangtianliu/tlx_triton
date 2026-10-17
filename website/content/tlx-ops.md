Production-ready kernels promoted from the TLX tutorials into the FBTriton op library.

## What tlx.ops is

- Promote performant and prod-impactful ops from TLX tutorials into the FBTriton op library.
- Catalog existing OSS TLX kernels and pick the best kernel per (arch, op).
- Enforce the highest quality bar for both code change and CI.
- Demote `tutorials/` to a playground scratch with no CI coverage.

## Interface

```python
from triton.tlx.ops import mm as tlx_mm

out = tlx_mm(a, b)
```

`tlx.ops` infers the architecture from the input tensors and dispatches to the
matching registered implementation.

## Structure

```
third_party/tlx/
    ops/                    -> (symlink) python/triton/tlx/ops
        kernels/
            mm/          sm90.py  sm100.py  gfx942.py  gfx950.py
            addmm/       gfx942.py
            bmm/         gfx942.py
            ...
```

## Current availability

Implemented today:

```
mm/sm100.py
flash_attn/sm100.py
flash_attn/gfx950.py
flash_attn_mxfp8/sm100.py
flash_attn_mxfp8/gfx950.py
hstu_attn/sm100.py
kda/sm100.py
```

Current kernel selections for the rest of the catalog:

```
kernels/
    mm/                   sm90.py  sm100.py  gfx942.py  gfx950.py
    addmm/                gfx942.py
    bmm/                  gfx942.py
    flash_attn/           sm90.py  sm100.py  gfx950.py  (fwd + bwd)
    flash_attn_mxfp8/     sm100.py  gfx950.py  (fwd + bwd)
    hstu_attn/            sm100.py  gfx942.py
                          _util.py  _stubs.py  _reference.py
    kda/                  sm100.py
    gdpa/                 sm100.py  gfx950.py
    grouped_gemm/         sm100.py  gfx950.py
    addmm_glu/            gfx950.py
    paged_decode/         gfx950.py
    ikbo_fa/              gfx950.py
    ikbo_lce/             gfx950.py
    multi_cta_layernorm/  sm100.py
    cross_attention/      sm100.py
    bmm_shared_a/         gfx950.py
```

`flash_attn_mxfp8` accepts contiguous BF16 Q/K/V of identical shape
`[batch, heads, sequence, 128]`, with sequence lengths divisible by 256.
On gfx950, each head must also contain fewer than `2**31` elements.
The gfx950 implementation supports causal and non-causal attention. Backward
uses native E4M3/E8M0 scaled products, fixed probability scale `2**-8`, and a
shared 32-by-32 dS scale for dQ and dK. Separate Q and KV owners recompute QK
and dP (seven executed products, five unique products), accumulate all gradients
in FP32, and store BF16 without atomics. Backward supports PyTorch deterministic
mode. The gfx950 forward uses 32-by-32 Q/K quantization, whereas the Blackwell
public wrapper uses 1-by-32 groups; both use E4M3 data and E8M0 scales.
