# gfx950 regression checkpoint

This branch preserves the local work on 2026-10-17.
Performance recovery remains incomplete.
The performance campaign remains paused.

The snapshot includes the existing BF16 backward changes, gfx950 MXFP8 source, tests, and operator documentation.
It also includes selected campaign source and evidence files.
It contains no LLVM source changes.

## Source and evidence

- The working kernel is [gfx950_bwd.py](../../../third_party/tlx/ops/kernels/flash_attn/gfx950_bwd.py).
- The tested frozen candidate is [accepted_gqa8_k3_gqa16_k5.py](frozen_sources/accepted_gqa8_k3_gqa16_k5.py).
- The deferred GQA16K candidate is [gqa16_phase1_early_Q_fragment_boundary01.py](frozen_sources/gqa16_phase1_early_Q_fragment_boundary01.py).
- The saved ten-case comparison is [all10_report.txt](evidence/all10_report.txt).
- The last pause record is [pause.json](evidence/pause.json).
- The archived file hashes and original paths are in [manifest.json](manifest.json).
- The earlier explanation is [PR2709_OPTIMIZATION_REPORT.html](../../../PR2709_OPTIMIZATION_REPORT.html).

The ten-case comparison uses frozen source images.
It does not validate the complete source tree in this commit.
GQA16K remains 3.858% below its historical matched throughput.
The historical and current median throughputs are 888.317 and 853.806 TFLOPS.
The report defines the paired comparison and measurement limits.

The HTML report predates this checkpoint.
Its embedded figures render without external resources.
Its 709 evidence links require the local `.campaign` directory.
That directory contains 380 GB of local artifacts and is excluded from this commit.
The selected archive does not replace the complete campaign directory.

## Pending experiments

The MHA1K instruction-sinking control reduces static scratch loads from nine to five.
It retains 16 private bytes, three compiler spill entries, and 512 register words.
This control has no measured performance result.

The saved VMEM closure passed 81 rows and five negative controls.
All six runtime constructor checks passed without GPU execution.
The replacement combiner and runtime binder still require an independent source review.
Their earlier review failed because a text filter omitted `.patch` files.
The replacement source corrects that filter.

The GQA16K partial-completion candidate still lacks a completed semantic proof.
The latest proof stopped at different numeric MMRA metadata identifiers with identical tag values.
No physical audit or GPU timing followed that failure.

Files in `pending_runtime` retain their original absolute path bindings.
They are archival copies and are not runnable from this directory.
Custom compiler stages, compiled images, caches, and raw traces remain local.

## Validation for this snapshot

Python syntax checks passed for all 13 changed or new production and test Python files.
Whitespace checks passed for source, tests, and checkpoint notes.
The generated HTML and archived patch retain their original whitespace.
The following CPU test command passed seven tests:

```bash
ROCR_VISIBLE_DEVICES='' HIP_VISIBLE_DEVICES='' PYTHONPATH=python \
  /home/bangtliu/meta_venv/bin/python -m pytest -s --tb=short \
  python/test/tlx_benchmark/test_harness.py \
  -k 'mxfp8_backward_check or operator_focus_suite_names_and_host_defaults_are_stable'
```

No GPU correctness test or performance test ran during this snapshot operation.
No native compiler rebuild ran because the committed changes contain no native source edits.

## Resume procedure

1. Review the replacement combiner and runtime binder.
2. Complete aggregation and validate the runtime bindings.
3. Detect a quiet window on physical GPU 7 at `0000:f5:00.0`.
4. Run the matched MHA1K experiment with the saved images.
5. Check correctness, image identity, and GPU ownership before interpreting timing.
6. Compare any successful control against the historical case before adoption.
7. Continue the GQA16K semantic proof and regression investigation.

Keep kernel and TLX changes within the existing scope.
Preserve native MFMA arithmetic.
Do not change LLVM source.
