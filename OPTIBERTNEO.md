# OptiBERTneo real-valued 1.3B-token H100 runbook

> For the current parameter-matched real/multispace pair, use
> `OPTIBERTNEO_MULTISPACE.md`. This page remains the legacy real-only runbook.

This runbook trains the real-valued OptiBERTneo short-run variant from the
[EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-main.1804/) on two
Slurm nodes with four H100 GPUs per node.

> **Name clarification:** `1.3B` is the number of scheduled training token
> positions, not the model size. The model has **198,225,408 non-embedding
> parameters** and **236,828,928 total unique parameters**. The paper rounds
> the former to 198M. This repository calls the model `real`; `baseline` is a
> legacy alias for the same model.

## Readiness status

The configuration, preprocessing job, read-only preflight, NCCL smoke test,
two-node launcher, checkpointing, and final Hugging Face export are prepared.
No H100 is visible from the current development session, so the CUDA build,
NCCL path, memory use, and end-to-end training job have **not** been validated
on the target hardware here. Do not skip the NCCL and two-step smoke gates
below.

Site-specific partition, account, QoS, reservation, constraint, and network
interface values are intentionally not hard-coded. Supply them to `sbatch`.

## Recipe represented by this checkout

### Paper-aligned settings

- 28 encoder layers, hidden width 768, and 12 attention heads of width 64
- ordinary real-valued attention in every layer
- pre-RMSNorm, embedding RMSNorm, RoPE, bias-free Transformer blocks, and
  SwiGLU with effective feed-forward branch width 2,048
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  and the
  [RoBERTa tokenizer](https://huggingface.co/FacebookAI/roberta-base), with
  vocabulary size 50,265
- sequence length 1,024 and padding-free packing with cross-document
  attention blocked
- 20% masked-language-model selection, with every selected token replaced by
  `<mask>`
- AdamW with peak learning rate `6e-4`, betas `(0.9, 0.95)`, weight decay
  `0.1`, and gradient clipping at `1.0`
- 500 warmup steps, followed by cosine decay to 10% of the peak learning rate
  at step 620
- global batch 2,048 sequences per optimizer step
- 620 optimizer steps:
  `620 * 2,048 * 1,024 = 1,300,234,240` scheduled token positions

The model YAML uses `intermediate_size: 3072` because NeoBERT applies the
standard two-thirds SwiGLU conversion; the actual SwiGLU branch width is
2,048.

### Reconstruction choices

The paper does not release enough information for a bitwise reproduction of
the checkpoint or exact data order. This checkout therefore pins and records
the choices needed for an executable reconstruction:

- FineWeb-Edu `sample-10BT` at immutable revision
  `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5`; documents are selected in
  source order until approximately 1.6B source tokens are available
- `FacebookAI/roberta-base` at commit
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- seed `0`, initialization ranges `0.02`, RMSNorm epsilon `1e-5`, AdamW
  epsilon `1e-8`, zero dropout, tied input/output embeddings, and no LM-head
  bias
- BF16 training with TF32 enabled, native SwiGLU, PyTorch FlexAttention,
  `torch.compile`, and ordinary PyTorch DDP
- deterministic preprocessing logic and an explicit dataset manifest, but
  not the paper authors' unreleased token order

The resulting non-embedding count differs from the paper's algebraic target
`12 * 768^2 * 28 = 198,180,864` by about `+0.0225%`; the difference is small
normalization state included by the implementation.

## Distributed batch arithmetic

The supplied Slurm job uses one Slurm task per node. Each task launches four
training processes, one per GPU:

| Quantity | Value |
| --- | ---: |
| Nodes | 2 |
| H100s per node | 4 |
| DDP world size | 8 |
| Microbatch per GPU | 32 sequences |
| Sequences per distributed micro-step | `8 * 32 = 256` |
| Gradient accumulation | 8 |
| Global batch | `256 * 8 = 2,048` sequences |
| Token positions per optimizer step | `2,048 * 1,024 = 2,097,152` |
| Optimizer steps | 620 |
| Dataset rows consumed | 1,269,760 |
| Scheduled token positions | 1,300,234,240 |

If 32 sequences per H100 does not fit, reduce `MICRO_BATCH` to a divisor of
256, such as 16 or 8. The launcher derives gradient accumulation as
`2048 / (8 * MICRO_BATCH)`, preserving the paper's global batch. Run the
two-step smoke again after changing it.

## 1. Build a clean SM90 environment

The expected checkout layout is:

```text
ComplexAttention/
├── pytorch/       # custom PyTorch source checkout
└── NeoBERT/       # this repository
```

Run the setup on a build node with outbound network access, Conda, enough
temporary storage for a PyTorch source build, and the site's CUDA 12.6 toolkit
loaded. Choose an environment path on a filesystem mounted identically on
both compute nodes.

```bash
cd /mnt/nfs/home/st171793/ComplexAttention/NeoBERT

# Use the equivalent command/name at your site.
module load CUDA/12.6.0

export OPTIBERT_ENV_PREFIX=/shared/path/optibertneo-h100
export OPTIBERT_BUILD_ROOT=/shared/scratch/$USER/optibertneo-h100-build
export MAX_JOBS=16

bash scripts/setup_optibertneo_h100.sh
export OPTIBERT_PYTHON="$OPTIBERT_ENV_PREFIX/bin/python"
```

The setup script:

1. refuses to alter an existing environment;
2. creates a Python 3.11 Conda environment;
3. builds the sibling PyTorch checkout with `TORCH_CUDA_ARCH_LIST=9.0`;
4. installs the exact Triton pin recorded by that PyTorch checkout;
5. installs the pinned training dependencies and both editable projects;
6. verifies the SM90 build and runs the dependency/runtime preflight; and
7. writes
   `$OPTIBERT_ENV_PREFIX/optibertneo-build-manifest.txt`.

If the setup was interrupted after creating the environment, choose a new
empty `OPTIBERT_ENV_PREFIX` for the retry. Do not silently reuse another
PyTorch environment: an SM86-only build cannot run kernels on H100.

In every new submission shell, reload CUDA and restore the shared paths:

```bash
cd /mnt/nfs/home/st171793/ComplexAttention/NeoBERT
module load CUDA/12.6.0

export OPTIBERT_PROJECT_ROOT="$PWD"
export OPTIBERT_ENV_PREFIX=/shared/path/optibertneo-h100
export OPTIBERT_PYTHON="$OPTIBERT_ENV_PREFIX/bin/python"
export OPTIBERT_DATASET=/shared/path/datasets/fineweb_edu_roberta_1p6b
export HF_HOME=/shared/path/cache/huggingface
```

The Slurm jobs inherit the loaded CUDA environment; they do not guess a site
module name.

## 2. Validate the resolved recipe

These checks do not require a GPU:

```bash
"$OPTIBERT_PYTHON" scripts/pretraining/preflight_optibertneo.py --config-only
"$OPTIBERT_PYTHON" scripts/pretraining/inspect_optibertneo.py baseline --check

DRY_RUN=1 \
NUM_MACHINES=2 \
GPUS_PER_NODE=4 \
EXPECTED_NUM_MACHINES=2 \
EXPECTED_WORLD_SIZE=8 \
OPTIBERT_DATASET="$OPTIBERT_DATASET" \
PYTHON_BIN="$OPTIBERT_PYTHON" \
bash jobs/optibertneo-1p3b.sh real
```

`inspect_optibertneo.py` retains the older name `baseline`; it inspects the
same real-valued model selected by `jobs/optibertneo-1p3b.sh real`.

The dry run must report:

```text
topology=2x4 world_size=8
micro_batch=32 gradient_accumulation=8
global_batch=2048 sequences (2097152 token positions)
steps=620 scheduled_tokens=1300234240
```

## 3. Prepare the padding-free dataset

Submit from the NeoBERT repository root. The data job requests one node,
64 CPUs, 256 GiB RAM, and 24 hours. It needs outbound access to Hugging Face,
so select a CPU/data-transfer partition that permits downloads.

```bash
sbatch \
  --partition=<cpu-or-data-partition> \
  --account=<slurm-account> \
  --export=ALL \
  jobs/slurm/prepare-optibertneo-data.sbatch
```

Replace the angle-bracket placeholders before running the command. Add the
site's `--qos`, `--reservation`, or proxy settings if required.

The preprocessing job:

- downloads the pinned FineWeb-Edu and RoBERTa revisions;
- tokenizes segments with their BOS/EOS tokens;
- concatenates real tokens into full 1,024-token rows;
- assigns a document ID to every token so attention cannot cross document
  boundaries;
- allows a document to continue into the next row;
- drops the single incomplete tail row; and
- saves the dataset atomically.

There are **no padding positions** in the saved rows. This supersedes older
descriptions of the reconstruction that allowed up to two pads.

The output must contain:

```text
$OPTIBERT_DATASET/
├── dataset_info.json
├── optibertneo_manifest.json
├── tokenizer/
└── ... Arrow shards ...
```

The job refuses to overwrite an existing dataset. If a manifest already
exists, it validates the dataset instead of rebuilding it. After the job
finishes, run the read-only validation explicitly:

```bash
"$OPTIBERT_PYTHON" scripts/pretraining/preflight_optibertneo.py \
  --dataset "$OPTIBERT_DATASET"
```

Do not proceed unless it confirms at least 1,269,760 rows of length 1,024,
the `input_ids` and `document_ids` columns, padding-free packing, and matching
tokenizer IDs.

## 4. Prove the two-node NCCL path

The smoke job requests exactly two nodes and four H100s per task/node, creates
eight ranks, performs an NCCL all-reduce, and prints the hostname, local rank,
GPU model, compute capability, and memory for every rank.

```bash
sbatch \
  --partition=<h100-partition> \
  --account=<slurm-account> \
  --export=ALL \
  jobs/slurm/optibertneo-nccl-smoke-2n8g.sbatch
```

Success ends with:

```text
NCCL smoke passed: world_size=8, rank_sum=28
```

Both H100 job files use `#SBATCH --gpus-per-task=h100:4`. If the cluster uses
another GRES convention, override it with the site's supported command-line
flags (for example a constraint plus four untyped GPUs) or adjust that
directive locally. Preserve two nodes, one Slurm task per node, and exactly
four visible H100s per task.

No network interface is hard-coded. If rendezvous or NCCL fails, use the
interface and fabric settings supplied by the cluster administrators; do not
guess `NCCL_SOCKET_IFNAME`, UCX devices, or InfiniBand policy.

## 5. Run a two-optimizer-step training smoke

This uses the production training job and therefore exercises the per-node
preflight, dataset loading, H100/SM90 checks, FlexAttention, compilation,
forward/backward, DDP, optimizer update, checkpoint save, and final export.
It keeps the per-GPU microbatch at 32 but uses one distributed micro-step per
optimizer step, for a small two-step test.

Use a disposable run directory that will never be reused for the full run:

```bash
export SMOKE_TEST=1
export WANDB_MODE=disabled
export RUN_ROOT="$PWD/logs/optibertneo-real-smoke-$(date +%Y%m%d-%H%M%S)"

sbatch \
  --partition=<h100-partition> \
  --account=<slurm-account> \
  --time=00:30:00 \
  --export=ALL \
  jobs/slurm/optibertneo-real-1p3b-2n8g.sbatch

unset SMOKE_TEST RUN_ROOT
```

The job must finish normally and create `final_model/training_summary.json`
under the smoke `RUN_ROOT`. Investigate every preflight failure or CUDA/NCCL
error before launching the full run.

## 6. Submit the full 620-step run

Choose a stable `RUN_ROOT` on shared storage. The same absolute path must be
visible on both nodes and reused on every resubmission:

```bash
unset SMOKE_TEST
export WANDB_MODE=offline
export RUN_ROOT=/shared/path/runs/optibertneo-real-1p3b

sbatch \
  --partition=<h100-partition> \
  --account=<slurm-account> \
  --time=04:00:00 \
  --export=ALL \
  jobs/slurm/optibertneo-real-1p3b-2n8g.sbatch
```

Add site-specific flags such as `--qos`, `--reservation`, or `--constraint`
on the `sbatch` command line. Increase the time limit if the smoke-test
throughput predicts more than four hours.

The defaults used by the job are:

```text
NUM_MACHINES=2
GPUS_PER_NODE=4
MICRO_BATCH=32
GLOBAL_SEQUENCES=2048
DATALOADER_WORKERS=8
TORCH_COMPILE=true
WANDB_MODE=offline
```

Optional operational overrides include:

- `STAGE_DATASET=1`: copy the prepared dataset independently to each node's
  `$SLURM_TMPDIR` before launch; ensure local scratch is large enough.
- `MICRO_BATCH=16` or `8`: reduce peak GPU memory while preserving the global
  batch through higher gradient accumulation.
- `WANDB_MODE=online` and `WANDB_ENTITY=<entity>`: enable online tracking when
  compute nodes have network access and credentials.
- `NCCL_DEBUG=INFO`: increase NCCL diagnostics for a failing run.

Do not change `GLOBAL_SEQUENCES`, the 620 steps, sequence length, masking
recipe, or scheduler if the goal is the paper-aligned 1.3B-token run.

## Resume after timeout or preemption

`trainer.resume` is enabled. Accelerator state is saved every 100 optimizer
steps, with the three newest training-state checkpoints retained. To resume,
reload the same environment, export the **same** `OPTIBERT_DATASET` and
`RUN_ROOT`, keep the same topology/batch settings, and submit the same full
job again.

The job file's fallback run directory includes `$SLURM_JOB_ID`; that default
is convenient for isolated attempts but cannot resume across different job
IDs. Explicitly exporting a stable `RUN_ROOT` avoids that trap.

Never run two jobs concurrently against the same `RUN_ROOT`, and never reuse
the two-step smoke directory for the full run.

## Outputs and completion checks

Slurm writes `optibertneo-real-1p3b-<jobid>.out` and `.err` in the submission
directory. The stable run directory contains:

```text
$RUN_ROOT/
├── checkpoints/             # resumable optimizer/scheduler/RNG/metric state
├── model_checkpoints/
│   └── <step>/
│       ├── _SUCCESS
│       └── state_dict.pt
├── hydra/                    # resolved Hydra run files
├── wandb/                    # offline or online W&B files
└── final_model/
    ├── ... tokenizer files ...
    ├── ... safetensors/config files ...
    └── training_summary.json
```

Model snapshots are written every 100 steps, and a final step-620 snapshot
and Hugging Face-compatible `final_model` export are written at normal
completion. Inspect the final accounting with:

```bash
"$OPTIBERT_PYTHON" -m json.tool \
  "$RUN_ROOT/final_model/training_summary.json"
```

For a complete run, `optimizer_steps` should be 620,
`training_sequences` should be 1,269,760, and `training_tokens` should be
1,300,234,240. Because packing is padding-free, training tokens and scheduled
token positions should agree exactly.

## Fast failure diagnosis

- **Preflight reports no SM90:** the wrong PyTorch runtime is being imported.
  Confirm `"$OPTIBERT_PYTHON" -c 'import torch; print(torch.__file__);
  print(torch.cuda.get_arch_list())'` includes `sm_90`.
- **Fewer than four devices on a node:** fix the site's GPU request/binding;
  do not weaken the expected topology checks.
- **Dataset manifest or row-count failure:** rebuild to a new empty
  `OPTIBERT_DATASET`; do not train from a partial or legacy padded dataset.
- **NCCL smoke hangs or reports only one node:** resolve Slurm rendezvous and
  fabric configuration before attempting training.
- **CUDA out of memory:** lower `MICRO_BATCH`, confirm the derived global batch
  remains 2,048, then repeat the two-step smoke.
- **Job starts from step zero after resubmission:** the `RUN_ROOT` changed or
  contains no complete Accelerator checkpoint.
