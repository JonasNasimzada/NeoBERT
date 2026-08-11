# Equal-parameter, equal-token attention-space ablation

This recipe trains seven homogeneous masked-language models on BabyLM 2026
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

All models otherwise use 6 layers, width 256, 8 heads, GELU, RoPE, pre-RMSNorm,
tied input/output embeddings, no bias in the MLM head, and no dropout. The
parameter equality is exact; the validator checks both the total and every
encoder layer before a sweep.

This is parameter and token matching, not wall-time or FLOP matching. The
real-valued attention projections use fewer parameters, so their models use the
wider 2,562-unit FFN to keep the layer and model totals identical.

## Packing contract

The preprocessing job concatenates tokenized documents and writes exact
512-token rows. It drops only the final incomplete tail, so training has no
padding or unpadding overhead. The same packed rows and deterministic split are
used by all seven models.

The rows intentionally do **not** contain `document_ids`. Direct Flash SDPA
cannot represent NeoBERT's block-diagonal document mask. Consequently all seven
runs allow attention across document boundaries inside a packed row. This is
the controlled seven-way comparison that includes `complex/flash` and
`real/flash` without changing the attention semantics for those runs. If
document boundaries must be isolated, use FlexAttention and treat that as a
separate experiment; do not label it `flash`.

## Training budget

The common training configuration is:

- BabyLM 2026 Strict, pinned in the dataset config;
- Google BERT uncased tokenizer, vocabulary 30,522, context 512;
- 20% MLM masking using NeoBERT's 100%-`[MASK]` corruption;
- micro-batch 8, gradient accumulation 4, effective batch 32;
- AdamW at `6e-4`;
- BF16 with TF32 enabled;
- 42,000 optimizer steps and exactly 688,128,000 presented token positions per
  model;
- 2,735 warmup steps, then cosine decay through step 42,000;
- a W&B training record every 420 steps and deterministic validation/checkpoint
  points every 7,000 steps.

One step always contains 32 packed sequences, or 16,384 token positions. All
seven variants therefore run the same number of updates and see the same token
budget. The 42,000-step ceiling puts the slowest backend from the completed A100
pilot near the three-hour allocation; faster backends finish earlier. Keeping a
finished fast job idle would not train it further or improve the comparison.

A 10,620-second emergency guard leaves about three minutes before Slurm
termination. If that guard fires before step 42,000, the W&B run is marked
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
export DATASET_PATH=/shared/path/babylm-2026-strict-bert512-flat
export HF_HOME=/shared/path/huggingface-cache
```

Prepare the deterministic train/validation `DatasetDict` once on CPU:

```bash
sbatch \
  --partition=slowlane \
  --qos=hiwi_project \
  jobs/attention_ablation/prepare_data.sbatch
```

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

Before the full run, launch a two-step seven-way A100 smoke array:

```bash
export SMOKE_RUNS_ROOT=/shared/path/complex-attention-ablation-smoke
sbatch \
  --partition=slowlane \
  --gpus=A100:1 \
  --qos=hiwi_project \
  --array=0-6 \
  --time=00:20:00 \
  --export=ALL,RUNS_ROOT="$SMOKE_RUNS_ROOT",EXPERIMENT_ID=smoke-v1,MAX_STEPS=2,WARMUP_STEPS=1,WANDB_MODE=disabled \
  jobs/attention_ablation/train.sbatch
```

To smoke-test only the two real-valued additions, use the same command with
`--array=5-6`.

Set W&B and output locations, then submit the training array and its correlated
benchmark array:

```bash
export RUNS_ROOT=/shared/path/complex-attention-ablation
export EXPERIMENT_ID=a100-3h-v1
export WANDB_PROJECT=complex-attention-ablation
export WANDB_ENTITY=hyper_attention
export WANDB_MODE=online

bash jobs/attention_ablation/submit.sh
```

The submission script uses the requested resource command for both arrays:

```text
sbatch --partition=slowlane --gpus=A100:1 --qos=hiwi_project --array=0-6 ...
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

The dependent benchmark job evaluates the final export at contexts 128, 256,
and 512 with deterministic masking and an equal token budget. It writes a JSON
report, logs all scalar metrics to a W&B `benchmark` run in the same group, and
uploads the report as an artifact.

For the official BabyLM zero-shot suite, prepare and pin a checkout of
`babylm-org/babylm-eval`, download its evaluation data (including any gated
assets), and export `BABYLM_EVAL_ROOT` before submission. The benchmark job will
run the MLM zero-shot script when that variable is present, then upload parsed
metrics and the complete official result directory to W&B. Keep the evaluator
commit in the run notes so scores remain reproducible.

The official harness pads variable-length MLM batches. For strict
`complex/flash` and `real/flash` checkpoints, a scoped evaluator-only adapter
runs each padded row at its true length and restores batch-shaped logits. This
preserves the selected Flash backend and avoids silently falling back to Torch
SDPA.

The `torch` variants permit PyTorch's normal SDPA selection and may themselves
choose a Flash kernel on A100; the `flash` variants strictly select PyTorch's
Flash SDPA backend. They do not require the external `flash-attn` package.
Treat the corresponding torch/flash pairs as equivalent implementations and
use the runtime/memory logs to distinguish dispatch behavior.
