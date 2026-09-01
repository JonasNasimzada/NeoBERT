# Real-MHA FineWeb-Edu A100 runbook

This is the exactly parameter-matched real-valued control for the multispace
FineWeb-Edu experiment. It uses the same tokenizer, packed 1,024-token dataset,
optimizer schedule, effective batch, seed convention, and A100 FlashAttention
runtime as the multispace run. Only the attention space and the compensating
feed-forward width differ.

The defining files are:

- `conf/model/attention-ablation-real-100m.yaml`
- `conf/dataset/fineweb_edu_google_1024.yaml`
- `scripts/attention_ablation/validate_100m_pair.py`
- `jobs/real_fineweb/train.sbatch`
- `jobs/real_fineweb/submit.sh`

## Exact model match

Both models have hidden width 768, nine encoder blocks, 12 heads of width 64,
a tied 30,522-token input/output embedding, GELU feed-forward layers, RMSNorm,
RoPE, and no linear biases. The real model uses ordinary real-valued MHA in
every head and every layer. Its Q, K, and V head outputs are concatenated and
mixed by one standard output projection.

| Quantity | Multispace | Real MHA |
| --- | ---: | ---: |
| Hidden size | 768 | 768 |
| Layers | 9 | 9 |
| Heads | 4 complex + 4 split-complex + 4 dual | 12 real |
| Head width | 64 | 64 |
| Attention parameters/block | 4,718,592 | 2,359,296 |
| GELU FFN width | 2,464 | 4,000 |
| FFN parameters/block | 3,784,704 | 6,144,000 |
| Two RMSNorms/block | 1,536 | 1,536 |
| Total parameters/block | 8,504,832 | 8,504,832 |
| Total trainable parameters | 99,985,152 | 99,985,152 |

The equality follows directly for hidden width \(H=768\), vocabulary
\(V=30{,}522\), and \(L=9\) layers:

```text
real attention       = 3H^2 + H^2       = 2,359,296
real FFN (I=4,000)   = 2HI              = 6,144,000
two block RMSNorms   = 2H               =     1,536
real block total                         = 8,504,832

multispace attention = 6H^2 + 2H^2       = 4,718,592
multispace FFN       = 2H(2,464)         = 3,784,704
two block RMSNorms   = 2H               =     1,536
multispace block total                   = 8,504,832

tied embedding + final RMSNorm = VH + H = 23,441,664
complete model = VH + H + L(8,504,832)  = 99,985,152
```

Run `scripts/attention_ablation/validate_100m_pair.py` only inside the requested
A100 environment. The training launcher does this automatically before every
segment. It constructs both models, checks the tied weights and attention
layouts, and verifies every block and total parameter count. Exact parameter
matching does not imply equal FLOPs, memory traffic, or wall-clock throughput;
those remain measured outcomes of the comparison.

## Shared FineWeb-Edu contract

The real control deliberately reuses the multispace dataset rather than
preprocessing a second copy. The source is the official
`HuggingFaceFW/fineweb-edu` `sample-10BT` subset at immutable revision
`87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`. Text is tokenized with the pinned
`google-bert/bert-base-uncased` tokenizer and packed into exact 1,024-token
rows.

The prepared directory is normally:

```text
tokenized_datasets/fineweb_edu_google_1024_1p6b/
├── dataset_dict.json
├── optibertneo_manifest.json
├── tokenizer/
├── train/
└── validation/
```

Rows contain only `input_ids: int32[1024]`. They have no padding mask or
`document_ids`; attention is bidirectional across document boundaries inside a
packed row. This is the same mask-free input contract used by multispace Flash.
Do not use `conf/dataset/fineweb_edu.yaml` or the legacy
`fineweb_edu_roberta_1p6b` cache: it uses a different 50,265-token RoBERTa
vocabulary and document-boundary masks.

## Identical training budget

| Quantity | Value |
| --- | ---: |
| Sequence length | 1,024 |
| Microbatch | 4 sequences |
| Gradient accumulation | 4 |
| Effective batch | 16 sequences/update |
| Token positions/update | 16,384 |
| Optimizer updates | 84,000 |
| Presented token positions | 1,376,256,000 |
| Warmup | 5,469 updates |
| LR decay | cosine through update 84,000 |
| Checkpoint and validation interval | 14,000 updates |
| Logging interval | 840 updates |

The token budget is exactly `84,000 * 4 * 4 * 1,024 = 1,376,256,000`
presented positions. Keep the product of microbatch and accumulation equal to
16 when comparing with multispace.

Training is split into resumable 15-hour allocations. The in-process guard is
53,640 seconds, leaving six minutes for a complete checkpoint. The default five
segments match the multispace continuation plan. Every later segment has an
`afterok` dependency on its predecessor and resumes only a checkpoint carrying
the completion marker.

## Environment and submission

Run from the NeoBERT repository root on the Slurm login node:

```bash
cd /mnt/nfs/home/st171793/ComplexAttention/NeoBERT

export DATASET_PATH="$PWD/tokenized_datasets/fineweb_edu_google_1024_1p6b"
export RUNS_ROOT="$PWD/logs/real_fineweb"
export EXPERIMENT_ID=fineweb-edu-s1024-real-100m-v1
export SEED=42

export WANDB_PROJECT=complex-attention-fineweb
export WANDB_ENTITY=hyper_attention
export WANDB_MODE=online
```

Keep `DATASET_PATH` identical for the paired runs. Keep `RUNS_ROOT`,
`EXPERIMENT_ID`, and `SEED` unchanged across continuations; changing any of
them creates a different checkpoint and W&B identity.

Inspect the exact Slurm commands without submission:

```bash
DRY_RUN=1 SKIP_PREP=1 bash jobs/real_fineweb/submit.sh
```

### Reuse a completed dataset

When both the DatasetDict and manifest already exist, launch only the five A100
training segments:

```bash
SKIP_PREP=1 bash jobs/real_fineweb/submit.sh
```

`SKIP_PREP=1` validates both required files before submission. The trainer then
checks the schema, fixed row length, tokenizer, split, and packing manifest.

### Follow an existing preparation job

To start the real control as soon as the same preparation job used by
multispace succeeds, attach its numeric Slurm ID:

```bash
PREP_JOB_ID=<preparation-job-id> bash jobs/real_fineweb/submit.sh
```

Do not combine `PREP_JOB_ID` with `SKIP_PREP=1`. The first real training segment
receives an `afterok` dependency on that preparation job, so both model runs
consume the same published dataset and a failed preparation cannot start
training.

### Prepare and train from scratch

With neither `PREP_JOB_ID` nor `SKIP_PREP=1`, the helper submits the shared
`jobs/multispace_fineweb/prepare_data.sbatch` recipe once and chains the five
real-MHA A100 segments after it:

```bash
bash jobs/real_fineweb/submit.sh
```

Preprocessing refuses to overwrite an existing output directory. Use
`SKIP_PREP=1` for a completed cache rather than trying to prepare it again.

### Resume an interrupted chain

Keep all identity variables unchanged and request additional continuation
segments:

```bash
SKIP_PREP=1 TRAIN_SEGMENTS=2 bash jobs/real_fineweb/submit.sh
```

The resumable state includes model, optimizer, scheduler, RNG, metrics, epoch,
and processed-batch position. Do not remove the latest complete
`checkpoints/checkpoint_*` directory.

## Completion check

The canonical paired-control directory is:

```text
$RUNS_ROOT/$EXPERIMENT_ID/real-flash/seed-$SEED/
```

Completion requires optimizer update 84,000 and
`train/completed_schedule=1`. Inspect:

```text
$RUNS_ROOT/$EXPERIMENT_ID/real-flash/seed-$SEED/final_model/training_summary.json
```

A final export written by an earlier wall-time segment is not by itself proof
of schedule completion; the persistent step metric and latest complete
Accelerate checkpoint are authoritative.
