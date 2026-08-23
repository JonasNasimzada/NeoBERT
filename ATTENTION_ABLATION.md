# Equal-parameter, equal-token attention-space ablation

This recipe trains twelve homogeneous masked-language models on BabyLM 2026
Strict. Every encoder layer in a model uses exactly one attention
space/backend pair.

| Array id | Attention space | Backend | FFN width | Trainable parameters |
| ---: | --- | --- | ---: | ---: |
| 0 | ordinary complex | native | 2,049 | 17,260,288 |
| 1 | ordinary complex | torch | 2,049 | 17,260,288 |
| 2 | ordinary complex | flash | 2,049 | 17,260,288 |
| 3 | split complex | native | 2,049 | 17,260,288 |
| 4 | split complex | torch | 2,049 | 17,260,288 |
| 5 | real | torch | 2,562 | 17,260,288 |
| 6 | real | flash | 2,562 | 17,260,288 |
| 7 | split complex | flash | 2,049 | 17,260,288 |
| 8 | dual number | native | 2,049 | 17,260,288 |
| 9 | dual number | torch | 2,049 | 17,260,288 |
| 10 | dual number | flash (hybrid tangent) | 2,049 | 17,260,288 |
| 11 | dual number | DFlash (`flash_fused`) | 2,049 | 17,260,288 |

Ids 0 through 6 remain unchanged so existing checkpoint paths and submitted
array tasks keep their original meaning. The split-Flash and dual variants are
appended at ids 7 through 11. Task 11 is the fused dual-number DFlash model; it
uses the same parameter-matched dual architecture and changes only the backend
to `flash_fused`.

All models otherwise use 6 layers, width 256, 8 heads, GELU, RoPE, pre-RMSNorm,
tied input/output embeddings, no bias in the MLM head, and no dropout. The
parameter equality is exact; the validator checks both the total and every
encoder layer before a sweep.

This is parameter and token matching, not wall-time or FLOP matching. The
real-valued attention projections use fewer parameters, so their models use the
wider 2,562-unit FFN to keep the layer and model totals identical.

## Packing contract

The preprocessing job concatenates tokenized documents and writes exact
1,024-token rows. It drops only the final incomplete tail, so training has no
padding or unpadding overhead. The same packed rows and deterministic split are
used by all twelve models.

The rows intentionally do **not** contain `document_ids`. Direct Flash SDPA
cannot represent NeoBERT's block-diagonal document mask. Consequently all twelve
runs allow attention across document boundaries inside a packed row. This is
the controlled twelve-way comparison that includes every direct-Flash variant
without changing the attention semantics for those runs. If
document boundaries must be isolated, use FlexAttention and treat that as a
separate experiment; do not label it `flash`.

## Training budget

The common training configuration is:

- BabyLM 2026 Strict, pinned in the dataset config;
- Google BERT uncased tokenizer, vocabulary 30,522, context 1,024;
- 20% MLM masking using NeoBERT's 100%-`[MASK]` corruption;
- micro-batch 4, gradient accumulation 4, effective batch 16;
- AdamW at `6e-4`;
- BF16 with TF32 enabled;
- 84,000 optimizer steps and exactly 1,376,256,000 presented token positions per
  model;
- 5,469 warmup steps, then cosine decay through step 84,000;
- a W&B training record every 840 steps and deterministic validation/checkpoint
  points every 14,000 steps.

One step always contains 16 packed sequences, or 16,384 token positions. All
twelve variants therefore run the same number of updates and see the same token
budget. Relative to the former 512-token setup, halving the sequence batch keeps
tokens per micro-batch and per optimizer step unchanged. The quadratic
attention work per presented token nevertheless grows at length 1,024, so the
previous six-hour calibration no longer applies. The dual-Flash hybrid and
fused DFlash paths are uncalibrated; the conservative fifteen-hour ceiling keeps
them comparable without changing the training schedule. Keeping a
finished fast job idle would not train it further or improve the comparison.

A 53,640-second emergency guard leaves about six minutes before Slurm
termination. If that guard fires before step 84,000, the W&B run is marked
incomplete and its dependent benchmark refuses to publish partial results.
Repeated passes over the packed BabyLM corpus are expected.

## One-time environment and data preparation

Use the custom PyTorch/ComplexAttention environment built for A100 (`sm_80`).
Do not use the repository's H100-only setup. Every batch job purges inherited
modules, loads `Miniconda3` and the cluster's default `CUDA` module, initializes
Conda through `$EBROOTMINICONDA3/bin/activate`, and activates `attention_dev`.
The environment name can be overridden with `CONDA_ENV_NAME`.

The jobs match normal interactive Conda activation, including packages already
available from your user site, and check all required imports before doing
expensive work. If that preflight reports missing packages, install the pinned
Python-side stack directly into the environment. PyTorch is deliberately not
part of this requirements file, so this keeps the custom
ComplexAttention-compatible PyTorch build:

```bash
module purge
module load Miniconda3
source "$EBROOTMINICONDA3/bin/activate"
conda activate attention_dev
PYTHONNOUSERSITE=1 python -m pip install -r requirements-optibertneo-h100.txt
```

Authenticate W&B once from the activated environment (or export
`WANDB_API_KEY` through Slurm):

```bash
python -m wandb login
```

The jobs expect these variables:

```bash
export CONDA_ENV_NAME=attention_dev
export COMPLEX_ATTENTION_ROOT=/mnt/nfs/home/st171793/ComplexAttention
export DATASET_PATH=/shared/path/babylm-2026-strict-bert1024-flat
export HF_HOME=/shared/path/huggingface-cache
```

The held-out MLM benchmark defaults to all 1,732,608 positions in the prepared
validation split (1,692 rows of 1,024), also exactly 423 complete 4,096-token
batches. Override it for a different prepared dataset with
`BENCHMARK_TOKEN_BUDGET`; the value must be divisible by 4,096 and by every
requested context length.

Prepare the deterministic train/validation `DatasetDict` once on CPU:

```bash
sbatch \
  --partition=slowlane \
  --qos=hiwi_project \
  jobs/attention_ablation/prepare_data.sbatch
```

The pinned source is the complete BabyLM 2026 Strict release: 100,000,000
source tokens in 11,601,896 rows. Preprocessing verifies that row count before
tokenization, applies no source-token limit, and then reserves 1% of source rows
for non-leaking validation. Thus the complete release is ingested, with 99% used
for optimization and 1% used only for held-out metrics. The actual validated
source-row count is stored in the manifest and checked again by every training
task. Packing discards only the final incomplete tail of fewer than 1,024
WordPiece positions in each split. With the pinned source and tokenizer this
produces 168,236 training rows and 1,692 validation rows.

The output path must not already exist; preprocessing refuses to overwrite it.
Data download needs a network-enabled job. Training can run with the Hub cache
offline afterward.

## Validate and submit

The CPU parameter check is safe to run before allocating a GPU:

```bash
python scripts/attention_ablation/validate_variants.py
```

The validator resolves the local NeoBERT and ComplexAttention source trees
itself; it does not require an editable NeoBERT installation or a manual
`PYTHONPATH`.

Before the full run, launch a two-step twelve-way A100 smoke array:

```bash
export SMOKE_RUNS_ROOT=/shared/path/complex-attention-ablation-smoke
sbatch \
  --partition=slowlane \
  --gpus=A100:1 \
  --qos=hiwi_project \
  --array=0-11 \
  --time=00:20:00 \
  --export=ALL,RUNS_ROOT="$SMOKE_RUNS_ROOT",EXPERIMENT_ID=smoke-s1024-v1,MAX_STEPS=2,WARMUP_STEPS=1,WANDB_MODE=disabled \
  jobs/attention_ablation/train.sbatch
```

To smoke-test only the fused dual-number DFlash model, use the same command with
`--array=11`. To test all dual-number implementations, use `--array=8-11`.

Set W&B and output locations, then submit the training array and its two
correlated benchmark arrays:

```bash
export RUNS_ROOT=/shared/path/complex-attention-ablation
export EXPERIMENT_ID=a100-s1024-1p376b-v1
export WANDB_PROJECT=complex-attention-ablation
export WANDB_ENTITY=hyper_attention
export WANDB_MODE=online

bash jobs/attention_ablation/submit.sh
```

The submission script uses the requested resource command for all arrays:

```text
sbatch --partition=slowlane --gpus=A100:1 --qos=hiwi_project --array=0-11 ...
```

Each training task first checks for an A100, SM80 code, BF16 support, and a
finite forward/backward step for its chosen backend. Outputs are written below
`$RUNS_ROOT/$EXPERIMENT_ID/`. Run roots and W&B IDs are stable by experiment,
variant, and seed rather than Slurm job id, so a requeued task resumes the same
checkpoint and W&B run. Changing `EXPERIMENT_ID` starts a clean experiment.
The submission helper clears inherited smoke schedule overrides before it calls
Slurm. Set `WANDB_MODE=offline` on isolated compute nodes and run `wandb sync`
later from a network-enabled node.

## Logged measurements

The training run logs resolved configuration, space/backend, exact parameter
counts, Slurm/GPU/runtime metadata, loss, pseudo-perplexity, masked accuracy,
learning rate, gradient/weight norms, cumulative samples/tokens, step time,
tokens/s, sequences/s, and peak CUDA memory. Validation corruption is reset to
the same seed on every evaluation.

The dependent model benchmark evaluates the final export at contexts 128, 256,
512, and 1,024 with deterministic masking and one fixed token budget per
context. The default 1,732,608 positions exhausts the prepared validation split
exactly and is divisible by every context and the 4,096-token batch budget.
It writes a JSON report, logs all scalar metrics to a W&B `benchmark` run in the
same group, and uploads the report as an artifact. Memory fields include the
post-warm-up model baseline, peak allocated and reserved allocator bytes, and
incremental inference workspace. These allocator measurements are not HBM
read/write traffic.

A separate dependent A100 array runs the raw-attention protocols from the two
FlashAttention papers. The FlashAttention-1 Appendix E.6 grid uses FP16 random
Q/K/V, batch 16, 8 heads of dimension 64, sequence lengths 128 through 65,536,
all padding-mask/dropout combinations, and 100 measurements. The
FlashAttention-2 Section 4.1 grid fixes 16,384 total tokens, uses hidden size
2,048 with head dimensions 64 and 128, tests causal and non-causal attention
from length 512 through 16,384, and takes 30 measurements. Projection layers
are excluded, matching the papers.

Every paper-benchmark row records forward, backward, and combined latency;
paper-normalized and algebra-aware TFLOP/s; input bytes; CUDA allocator
baseline, peak, reserved, and incremental workspace bytes; and an explicit
`ok`, `unsupported`, `oom`, or `error` status. Results are checkpointed to
`benchmarks/attention_papers.json`, streamed to a dedicated W&B benchmark run,
and uploaded as a W&B table and artifact. The Slurm job has a six-hour limit and
resumes a compatible partial JSON at its first unfinished row, preserving the
completed prefix and continuing at the matching W&B step. It refuses to combine
results when the case grid, seed, device/software identity, or W&B destination
has changed. Strict Flash rows that cannot express the paper's padding mask are
marked unsupported instead of silently switching backend. CUDA allocator peaks
are memory footprint measurements, not the HBM traffic counter reported by
profiler-based figures in the first paper.

Split-complex Flash is one strict packed Flash SDPA call for both idempotent
channels. Dual-number Flash is deliberately reported as a hybrid: strict Flash
computes the primal, while an exact dense analytic calculation computes the
tangent. Its measured time and peak memory include that dense tangent, so it
does not have FlashAttention's linear-memory scaling and long rows may report
OOM. In contrast, dual DFlash (`flash_fused`) streams both primal and tangent
through the fused Triton kernel and is the linear-attention-memory comparison
against `dual-native`. The FA1 padding rows and split/dual Flash dropout rows
are explicitly `unsupported`, never relabeled fallbacks.

For the official BabyLM zero-shot suite, prepare and pin a checkout of
`babylm-org/babylm-eval`, download its evaluation data (including any gated
assets), and export `BABYLM_EVAL_ROOT` before submission. The benchmark job will
run the MLM zero-shot script when that variable is present, then upload parsed
metrics and the complete official result directory to W&B. Keep the evaluator
commit in the run notes so scores remain reproducible.

The official evaluator is wrapped by a process-tree monitor. Its sampled peak
GPU process memory and peak host RSS are stored in JSON, logged below
`benchmark/babylm/system/`, and included in the W&B artifact. The sampling
interval defaults to 0.25 seconds and can be set with
`BENCHMARK_MEMORY_SAMPLE_INTERVAL`.

The official harness pads variable-length MLM batches. For strict
Flash checkpoints, a scoped evaluator-only adapter
runs each padded row at its true length and restores batch-shaped logits. This
preserves the selected Flash backend and avoids silently falling back to Torch
SDPA.

The `torch` variants permit PyTorch's normal SDPA selection and may themselves
choose a Flash kernel on A100; the `flash` variants strictly select PyTorch's
Flash SDPA backend for their Flash-capable core. They do not require the
external `flash-attn` package. The dual-Flash qualifier above is essential:
only the hybrid backend's primal is Flash, while its tangent remains dense;
`flash_fused` is the separate fused DFlash implementation.
