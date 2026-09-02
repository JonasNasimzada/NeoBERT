# Parameter-matched 200M and 300M attention models

This experiment provides two exactly parameter-matched model pairs: a
real-valued multi-head-attention baseline and a multispace model whose every
layer contains ordinary-complex, split-complex, and dual-number attention.
The configurations and launchers are implemented, but the commands below must
still be run; this document does not imply that preflight or training passed.

## Architecture

All four models use width 768, 12 heads of dimension 64, RoPE, RMSNorm, GELU,
zero dropout, bias-free linear layers, tied input/output embeddings, a
30,522-token vocabulary, and direct FlashAttention. Scaling from the existing
100M pair changes only depth.

| Scale | Attention | Layers | Heads per layer | FFN width | Exact parameters |
|---|---|---:|---|---:|---:|
| 200M | real MHA | 21 | 12 real | 4,000 | 202,043,136 |
| 200M | multispace MHA | 21 | 4 ordinary-complex + 4 split-complex + 4 dual | 2,464 | 202,043,136 |
| 300M | real MHA | 33 | 12 real | 4,000 | 304,101,120 |
| 300M | multispace MHA | 33 | 4 ordinary-complex + 4 split-complex + 4 dual | 2,464 | 304,101,120 |

The model YAMLs are under `conf/model/` as
`attention-ablation-{real,multispace}-{200m,300m}.yaml`.

### How one multispace layer combines its heads

The operation follows ordinary real-space MHA structurally:

1. One packed projection maps the real residual stream from `H` to `6H`. It
   creates two real components for Q, K, and V and divides the 12 heads into
   three equal groups.
2. Four heads run ordinary-complex attention, four run split-complex attention,
   and four run dual-number attention. Each head retains both output
   components.
3. The six component groups are concatenated into a `2H`-wide tensor.
4. One shared `2H -> H` output projection mixes all heads, components, and
   spaces back into the real residual stream.

There is no averaging, gating, or separate per-space readout. Real MHA likewise
concatenates all 12 real head outputs and applies one shared `H -> H` output
projection.

### Exact parameter matching

Let `H = 768`, vocabulary size `V = 30,522`, and depth `L` be 21 or 33.
The real attention block has `4H^2 = 2,359,296` projection parameters. The
multispace block has `6H^2` in its packed QKV projection plus `2H^2` in its
shared output projection, or `8H^2 = 4,718,592`.

The GELU FFN has `2HI` parameters. Its width is 4,000 for real MHA and 2,464
for multispace MHA:

```text
real:       4H^2 + 2H(4000) + 2H = 8,504,832 parameters/block
multispace: 8H^2 + 2H(2464) + 2H = 8,504,832 parameters/block
```

The multispace attention overhead is therefore exactly offset by its narrower
FFN. With a tied decoder, the shared embedding and final norm contribute
`VH + H = 23,441,664` unique trainable real scalars:

```text
total(L) = 23,441,664 + L * 8,504,832
total(21) = 202,043,136
total(33) = 304,101,120
```

This controls parameter count, depth, width, head count, and training tokens.
It does not make attention FLOPs, memory use, or throughput identical, so those
should be reported with downstream quality.

## A100 preflight

First inspect the complete submission graph. The launcher is dry-run by
default and does not submit anything:

```bash
bash jobs/scaled_fineweb/submit.sh all
```

Submit only the four-task validation array (up to four A100s in parallel):

```bash
DRY_RUN=0 bash jobs/scaled_fineweb/submit.sh preflight
```

Each array task owns one model. Together they validate all four complete graphs
and exact counts, run the focused tests inside an A100 allocation, perform one
BF16 forward/backward/AdamW step at full depth with sequence length 64, and
exercise every model's FlashAttention route at sequence length 1,024 with one
layer. The array does not load FineWeb-Edu, save a checkpoint, or submit
training. Reports are written to
`logs/scaled_fineweb/preflight-${SLURM_ARRAY_JOB_ID}/`.

For a more expensive full-depth, 1,024-token smoke step on all four models:

```bash
FULL_PRODUCTION_GEOMETRY=1 DRY_RUN=0 \
  bash jobs/scaled_fineweb/submit.sh preflight
```

Do not launch training until the preflight Slurm job exits successfully and its
JSON reports show `"status": "passed"`.

### HoreKa A100 preflight

`jobs/scaled_fineweb/preflight-horeka-a100.sbatch` is the HoreKa Green wrapper.
It uses account `hk-project-pai00130`, partition `accelerated`, one A100 per
array task, the `attention_dev` micromamba environment, and HoreKa's node-local
`$TMPDIR` for compiler caches. To run the four model checks only after Slurm job
5125968 succeeds:

```bash
cd /hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention/NeoBERT
jid=$(sbatch --parsable \
  --dependency=afterok:5125968 \
  --kill-on-invalid-dep=yes \
  jobs/scaled_fineweb/preflight-horeka-a100.sbatch)
jid=${jid%%;*}
echo "$jid"
```

Keep `FULL_PRODUCTION_GEOMETRY=0` for the first A100-40 validation. The default
still tests every full-depth graph at length 64 and every length-1,024 kernel at
one layer. Use `afterany` instead of `afterok` only if validation should run even
when job 5125968 fails.

## FineWeb-Edu training

The controlled default schedule is identical for all four models:

- dataset configuration `fineweb_edu_google_1024` with tokenizer
  `google-1024`;
- sequence length 1,024;
- micro-batch 2 and gradient accumulation 8, for 16 sequences/update;
- 84,000 optimizer steps, or 1,376,256,000 scheduled token positions;
- seed 42;
- five sequential, resumable 15-hour Slurm segments per model by default.

The four model chains can run in parallel after their shared A100 preflight and
dataset-preparation dependencies pass. A live submission requires an explicit
confirmation guard:

```bash
DRY_RUN=0 CONFIRM_FULL_SUBMISSION=YES \
  bash jobs/scaled_fineweb/submit.sh all
```

The generic launcher also supports scheduling preparation of the repository's
1.6B-token FineWeb-Edu dataset. To reuse an already prepared and validated
DatasetDict:

```bash
DATASET_PATH=/absolute/path/to/fineweb_edu_google_1024_1p6b \
SKIP_PREP=1 DRY_RUN=0 CONFIRM_FULL_SUBMISSION=YES \
  bash jobs/scaled_fineweb/submit.sh all
```

Selections can narrow a launch to `200m`, `300m`, `real`, `multispace`, or one
model such as `multispace-300m`. Useful overrides include `TRAIN_SEGMENTS`,
`MAX_STEPS`, `SEED`, `RUNS_ROOT`, and `PREFLIGHT_JOB_ID`. Keep
`MICRO_BATCH * GRAD_ACCUM = 16` for equal-batch comparisons, and keep the same
`MAX_STEPS`, data order, and seed across each real/multispace pair.

### Start training on HoreKa

After the preflight wrapper and prepared DatasetDict are present in the HoreKa
checkout, use the guarded launcher below. It submits the preflight array after
job `5125968`, then starts four parallel, resumable training chains only after
that array succeeds:

```bash
cd /hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention/NeoBERT
DATASET_PATH=/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention/NeoBERT/tokenized_datasets/fineweb_edu_google_1024_1p6b \
DRY_RUN=0 CONFIRM_FULL_SUBMISSION=YES \
  bash jobs/scaled_fineweb/submit-horeka-training.sh all
```

The launcher uses `attention_dev`, CUDA 12.9, one A100 per model, and defaults
to micro-batch 1 with gradient accumulation 16 for the 40-GB GPUs. Each
segment has a two-day Slurm allocation but stops at 169,200 seconds to save a
checkpoint before the wall-time limit; `TRAIN_SEGMENTS=5` is the default. Use
`DRY_RUN=1` first to inspect dependencies and paths. The training launcher
requests the full-context smoke by default; set `FULL_PRODUCTION_GEOMETRY=0`
for the shorter gate. If job `5125968` has already finished, check its state
with `sacct`; if it succeeded, set `UPSTREAM_JOB_ID=` before submitting,
because old job IDs can expire from Slurm's dependency window.

### If the FineWeb-Edu directory is incomplete

The HoreKa launcher deliberately refuses to start training unless both
`dataset_dict.json` and `optibertneo_manifest.json` exist. Prepare the pinned
dataset on a CPU node with:

```bash
cd /hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention/NeoBERT
DATASET_PATH=/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention/NeoBERT/tokenized_datasets/fineweb_edu_google_1024_1p6b \
  sbatch --parsable jobs/scaled_fineweb/prepare-horeka.sbatch
```

The preparation job never overwrites an existing directory. If that path is a
partial failed preparation, choose a new `DATASET_PATH` or inspect it and
remove it manually only when it is disposable. Once preparation finishes
successfully, rerun the training command. You can also queue training
immediately behind a preparation job by passing its ID as `PREP_JOB_ID`; the
launcher will wait for both preparation and preflight before starting any
A100 work. If preflight array `5128653` has succeeded, reuse it instead of
submitting another one:

```bash
UPSTREAM_JOB_ID= \
PREP_JOB_ID=<preparation-job-id> \
PREFLIGHT_JOB_ID=5128653 \
DATASET_PATH=/hkfs/home/project/hk-project-pai00012/st_st171793/ComplexAttention/NeoBERT/tokenized_datasets/fineweb_edu_google_1024_1p6b \
DRY_RUN=0 CONFIRM_FULL_SUBMISSION=YES \
  bash jobs/scaled_fineweb/submit-horeka-training.sh all
```
