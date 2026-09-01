# Multispace FineWeb-Edu A100 runbook

This runbook trains the approximately 100M-parameter multispace masked-language
model on the official FineWeb-Edu `sample-10BT` subset. It is a single-A100,
1,024-token continuation workflow: several 15-hour Slurm jobs share one run
directory, checkpoint stream, scheduler state, data position, and W&B run.

The files that define the recipe are:

- `conf/dataset/fineweb_edu_google_1024.yaml`
- `conf/tokenizer/google-1024.yaml`
- `conf/model/attention-ablation-multispace.yaml`
- `jobs/multispace_fineweb/prepare_data.sbatch`
- `jobs/multispace_fineweb/train.sbatch`
- `jobs/multispace_fineweb/submit.sh`

Do not substitute `conf/dataset/fineweb_edu.yaml` or its
`fineweb_edu_roberta_1p6b` cache. That older OptiBERTneo recipe has a different
token vocabulary and attention-mask contract, as explained below.

## Exact data contract

The source is
[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
configuration `sample-10BT`, split `train`, pinned to the immutable commit:

```text
87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
```

This is the public, ungated v1.4.0 revision. `sample-10BT` contains about 10B
GPT-2-tokenizer tokens, 9,672,101 documents, and 28.52 GB of parquet files.
The preprocessing recipe deliberately selects documents in source order until
their FineWeb `token_count` values reach approximately 1.6B. That source field
uses the GPT-2 tokenizer and is only a selection estimate; the training budget
is computed from the final Google BERT token IDs.

The tokenizer is `google-bert/bert-base-uncased`, vocabulary 30,522, pinned to:

```text
86b5e0934494bd15c9632b12f734a8a67f723594
```

Long documents are emitted as 1,024-token tokenizer chunks. The selected source
documents are split before tokenization into deterministic 99% train and 1%
validation partitions with seed 0, preventing chunks from one source document
from entering both partitions.

Within each split, preprocessing concatenates tokenized segments into exact
1,024-token rows and drops only that split's final incomplete tail. Every saved
row therefore contains only:

```text
input_ids: int32[1024]
```

There is no padding, `attention_mask`, or `document_ids` column. Attention is
bidirectional across document boundaries within a packed row. Tokenizer boundary
tokens remain in the stream, but they do not create a block-diagonal attention
mask. The manifest must report:

```text
packing.padding_free=true
packing.cross_document_attention=true
packing.document_ids=false
```

Here, **mask-free** means no padding or document-boundary attention mask. This
does not disable the MLM objective: the collator still selects 20% of eligible
tokens and replaces every selected token with BERT's `[MASK]` token.

Preprocessing refuses to publish fewer than 1,344,000 training rows. It writes
the dataset atomically and refuses to overwrite an existing output directory.
The expected layout is:

```text
tokenized_datasets/fineweb_edu_google_1024_1p6b/
├── dataset_dict.json
├── optibertneo_manifest.json
├── tokenizer/
├── train/
└── validation/
```

## Model and training budget

Every one of the nine encoder layers has 12 algebra-valued heads: four
ordinary-complex, four split-complex, and four dual-number heads. Their two
scalar components are concatenated and mixed by the layer's one shared output
projection. The model has 99,985,152 trainable scalar parameters and uses the
multispace Flash backend on an A100.

The fixed schedule is:

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
| W&B logging interval | 840 updates |

The token total is exact:

```text
84,000 * 4 * 4 * 1,024 = 1,376,256,000
```

It counts presented packed positions, not unique source tokens or MLM prediction
targets. Do not change microbatch and accumulation independently: their product
must remain 16 for this schedule.

## Storage and time planning

Plan storage before submission. The important quantities are:

| Item | Expected storage |
| --- | ---: |
| FineWeb-Edu `sample-10BT` parquet | 28.52 GB |
| Decoded source dataset reported by the HF Viewer | about 48.12 GB |
| Minimum required packed train IDs alone | about 5.1 GiB |
| Retained training checkpoints and final export | about 6.7 GiB |

The Hugging Face cache can temporarily contain downloaded parquet and decoded
Arrow data while the atomic dataset writer also holds its temporary output.
Allow **80–120 GB of peak preparation space**, plus about **10 GB** under the
run root for checkpoints, exports, and logs. If cache, prepared data, and runs
share one filesystem, reserve roughly 130 GB rather than adding only the final
directory sizes. The preparation job requests 64 CPUs, 256 GiB RAM, and up to
24 hours; actual duration depends strongly on Hugging Face bandwidth and cache
state.

Observed A100 smoke throughput is roughly 2.26–2.5 seconds per optimizer
update. Pure update time for 84,000 steps is therefore about 52.7–58.3 hours;
validation, checkpointing, startup, and export make **55–62 hours** a safer
end-to-end estimate.

Each training allocation is 15 hours, with the trainer guard set to 53,640
seconds (14 hours 54 minutes). The remaining six minutes are reserved for a
complete resumable checkpoint. Four guard windows provide 59.6 trainer hours
and are expected to be sufficient near the measured rate, but leave little
margin. The submission helper therefore defaults to `TRAIN_SEGMENTS=5`. If
training reaches update 84,000 during segment four, segment five resumes the
completed state and performs no further optimizer updates; it exits after the
short final validation/export path. Set `TRAIN_SEGMENTS=4` only after local
calibration shows the full schedule, including overhead, fits within 59.6
trainer hours.

## Environment and paths

Run commands from the NeoBERT repository root on the Slurm login node:

```bash
cd /mnt/nfs/home/st171793/ComplexAttention/NeoBERT

export DATASET_PATH="$PWD/tokenized_datasets/fineweb_edu_google_1024_1p6b"
export RUNS_ROOT="$PWD/logs/multispace_fineweb"
export EXPERIMENT_ID=fineweb-edu-s1024-multispace-100m-v1
export SEED=42

# Put the large reusable Hub cache on a filesystem with sufficient capacity.
export HF_HOME=/shared/path/huggingface-cache

export WANDB_PROJECT=complex-attention-fineweb
export WANDB_ENTITY=hyper_attention
export WANDB_MODE=online
```

Replace `/shared/path/huggingface-cache` with a real shared location visible to
the preparation node. The jobs use the A100 `attention_dev` runtime through
`jobs/attention_ablation/common.sh`; `CONDA_ENV_NAME` can override that
environment name. Compute nodes need access to the prepared dataset, run root,
and environment. The preparation node additionally needs outbound access to
Hugging Face.

Changing `EXPERIMENT_ID`, `RUNS_ROOT`, or `SEED` creates a different checkpoint
and W&B identity. Keep all three unchanged across continuation submissions.
Never point a fresh geometry or changed recipe at an older run root.

## Recommended submission

First inspect the exact Slurm commands without submitting anything:

```bash
DRY_RUN=1 bash jobs/multispace_fineweb/submit.sh
```

For a new dataset and run, submit the complete dependency chain:

```bash
bash jobs/multispace_fineweb/submit.sh
```

The helper submits one CPU preparation job, followed by five single-A100 jobs.
Every training segment has an `afterok` dependency on the preceding job. A
failed preparation or training segment therefore stops the remaining chain
instead of training from incomplete state.

The printed output records the preparation job ID, every training job ID, the
dataset path, and the experiment root. Save it with the experiment notes. Slurm
output appears in files named `multispace-fineweb-prep-<job>.out` and
`ca-fineweb-ms-train-<job>.out` unless site flags override them.

### Use an already prepared dataset

When `DATASET_PATH` already contains the complete DatasetDict and manifest, do
not submit preprocessing again: the atomic writer intentionally refuses to
overwrite it.

```bash
SKIP_PREP=1 bash jobs/multispace_fineweb/submit.sh
```

`SKIP_PREP=1` checks for both `dataset_dict.json` and
`optibertneo_manifest.json`. The training process then performs the full schema,
row-length, split, manifest, tokenizer, and packing validation before beginning
the training loop. The segment's small A100 runtime smoke runs first.

### Attach training to a separately submitted preparation job

This is useful when the data-transfer partition differs from the default
`slowlane` partition:

```bash
export NEOBERT_ROOT="$PWD"
export COMPLEX_ATTENTION_ROOT="$(cd .. && pwd)"

prep_result=$(sbatch --parsable \
  --partition=<data-transfer-partition> \
  --chdir="$NEOBERT_ROOT" \
  --export="ALL,NEOBERT_ROOT=$NEOBERT_ROOT,COMPLEX_ATTENTION_ROOT=$COMPLEX_ATTENTION_ROOT,DATASET_PATH=$DATASET_PATH" \
  jobs/multispace_fineweb/prepare_data.sbatch)
export PREP_JOB_ID="${prep_result%%;*}"

bash jobs/multispace_fineweb/submit.sh
```

Replace the angle-bracket partition placeholder. `PREP_JOB_ID` must be numeric.
Do not combine it with `SKIP_PREP=1`.

### Resume after a failed or exhausted chain

Keep the same `DATASET_PATH`, `RUNS_ROOT`, `EXPERIMENT_ID`, and `SEED`, then
submit additional continuation segments:

```bash
SKIP_PREP=1 TRAIN_SEGMENTS=2 bash jobs/multispace_fineweb/submit.sh
```

Each segment discovers the latest checkpoint carrying a `_SUCCESS` marker.
Incomplete checkpoint directories are retained for diagnosis but never loaded.
The resumable state includes model, optimizer, scheduler, metrics, RNG state,
epoch, and processed-batch position. A max-time stop saves a complete state and
exits normally so the next `afterok` segment can continue. Do not delete the
latest complete `checkpoints/checkpoint_*` directory between segments.

## Why the legacy FineWeb/RoBERTa cache is incompatible

The existing `conf/dataset/fineweb_edu.yaml` and
`tokenized_datasets/fineweb_edu_roberta_1p6b` belong to the separate
OptiBERTneo reconstruction. They cannot be reused for this multispace run:

1. **Wrong token IDs and vocabulary.** The legacy cache uses
   `FacebookAI/roberta-base` with 50,265 IDs. This model's tied embedding and
   decoder use the 30,522-entry Google BERT vocabulary. RoBERTa IDs can exceed
   the embedding range, and even shared numeric IDs do not denote the same
   tokens.
2. **Unsupported document mask.** The legacy preprocessing default writes an
   `input_ids` plus `document_ids` pair and uses the IDs to block attention
   across packed documents. Direct multispace FlashAttention cannot express
   NeoBERT's arbitrary block-diagonal document mask and explicitly rejects
   `document_ids`.
3. **Different attention semantics.** The new controlled recipe deliberately
   allows cross-document attention and saves only `input_ids`; silently
   discarding the legacy IDs would change the meaning of an already prepared
   dataset without recording that change in its manifest.
4. **Different provenance.** The legacy source config pins `v1.0.0`; this run
   pins the immutable v1.4.0 commit shown above.

The new manifest validator requires exactly `train` and `validation` splits,
only the `input_ids` column, fixed 1,024-token rows, BERT special-token IDs,
padding-free packing, cross-document attention, and `document_ids=false`.
These checks are intentional. Re-tokenize from the pinned FineWeb-Edu source;
do not rename, convert, or partially reuse the legacy cache.

## Completion checks

The canonical run directory is:

```text
$RUNS_ROOT/$EXPERIMENT_ID/multispace-flash/seed-42/
```

A successful schedule records update 84,000 and
`train/completed_schedule=1`. The final Hugging Face export and summary are:

```text
$RUNS_ROOT/$EXPERIMENT_ID/multispace-flash/seed-42/final_model/
└── training_summary.json
```

Confirm in `training_summary.json` that the schedule completed and did not end
only because of the wall-time guard. A partial export from an earlier segment
does not establish completion; the persistent optimizer-step metric and latest
complete Accelerate checkpoint are authoritative.
