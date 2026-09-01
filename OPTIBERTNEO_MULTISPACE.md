# Parameter-matched real and multispace OptiBERTneo

This runbook defines a controlled pair of 1.3B-token-position OptiBERTneo
experiments based on the [EMNLP 2025 paper](https://aclanthology.org/2025.emnlp-main.1804/)
([local PDF](</mnt/nfs/home/st171793/.codex/attachments/46566600-f20e-4bfb-85fe-2712f308f95b/2025.emnlp-main.1804-2.pdf>)).
The two models have the same depth, hidden size, head count, vocabulary,
training-token budget, and exact parameter count. They differ only in how the
attention/FFN parameter budget is divided.

> **Submission status:** the implementation and commands are prepared, but no
> FineWeb-Edu preprocessing job and no final 620-step training job were
> submitted. Commands under **Future cluster execution** are instructions, not
> a record of executed jobs.

## The paired architecture

Both models use 28 bidirectional encoder blocks, hidden size `H=768`, 12 heads
of width 64, pre-RMSNorm, embedding RMSNorm, RoPE, bias-free projections,
zero dropout, and SwiGLU. Every block in a model uses the same attention type;
there is no alternating layer schedule.

| Per block | Real OptiBERTneo | Multispace OptiBERTneo |
| --- | ---: | ---: |
| Attention heads | 12 real | 4 complex + 4 split-complex + 4 dual |
| Attention matrices | `4H² = 2,359,296` | `8H² = 4,718,592` |
| YAML `intermediate_size` | 3,072 | 1,536 |
| Effective SwiGLU width | 2,048 | 1,024 |
| FFN matrices | `3H(2048) = 4,718,592` | `3H(1024) = 2,359,296` |
| Two RMSNorm vectors | 1,536 | 1,536 |
| Exact block total | 7,079,424 | 7,079,424 |

NeoBERT converts the nominal YAML intermediate size to two thirds of that
width for SwiGLU. Halving the multispace FFN width therefore transfers exactly
`2,359,296` matrix parameters from the FFN to attention and keeps every block
parameter-matched.

### How one multispace layer combines its heads

The combination follows ordinary real multi-head attention: run independent
heads, concatenate every head output, then apply one shared output projection.
The spaces are not summed, averaged, gated, or collapsed before that
projection.

```text
pre-RMSNorm(x)
      |
      +-- shared packed QKV projection: 768 -> 4608 (= 6H)
              |
              +-- 4 complex heads       -> component 0, component 1
              +-- 4 split-complex heads -> component 0, component 1
              +-- 4 dual-number heads   -> primal, dual
                                   |
              concatenate all six 4x64 groups -> 1536 (= 2H)
                                   |
                     shared real output projection: 1536 -> 768
                                   |
                            residual connection
                                   |
                    pre-RMSNorm -> SwiGLU -> residual
```

The packed QKV matrix has shape `H x 6H`: each algebra-valued Q, K, and V has
two real scalar components. It is partitioned into three equal space groups,
each with four 64-wide heads. RoPE is applied to both Q/K components, and the
same document-boundary FlexAttention block mask is passed to all three space
kernels. Each kernel returns both components for each of its four heads. Their
flattened order is complex component 0/1, split-complex component 0/1, then
dual primal/dual. Concatenating those outputs produces `2H=1536` channels,
which one bias-free `2H -> H` projection mixes across every head and space.

This is the direct multispace analogue of real MHA, where `H -> 3H` produces
Q/K/V, 12 real head outputs concatenate back to `H`, and one shared `H -> H`
projection mixes them.

## What “198M” means

The paper's rounded 198M size is the non-embedding matrix budget

```text
12 * H² * L = 12 * 768² * 28 = 198,180,864.
```

It excludes the token embedding and normalization vectors. For both models in
this repository, the exact unique trainable counts are:

| Count | Real | Multispace |
| --- | ---: | ---: |
| Paper-style block matrices | 198,180,864 | 198,180,864 |
| Non-embedding, including all RMSNorm vectors | 198,225,408 | 198,225,408 |
| Tied RoBERTa token embedding (`50,265 x 768`) | 38,603,520 | 38,603,520 |
| **Total unique parameters** | **236,828,928** | **236,828,928** |

The decoder reuses the input embedding, so it is counted once. Thus this is a
faithful 198M *non-embedding* OptiBERTneo pair, not a 198M-total-parameter
pair. `scripts/pretraining/validate_optibertneo_pair.py` constructs both module
graphs on the meta device and verifies every count above.

Equal parameters and equal training tokens do **not** imply equal FLOPs or
throughput. Multispace attention has twice as many real projection parameters,
two-component algebra arithmetic, and a narrower FFN. After the shared QKV
projection, dual and split attention run on two persistent side streams while
ordinary-complex attention remains on the caller's current stream. Every layer
joins both side streams before concatenation and reuses the same two-stream
pool for the next layer. Set `multispace_cuda_streams: false` for a serial GPU
ablation. CUDA streams permit overlap but do not guarantee it, so report wall
time, tokens/s, peak memory, and loss alongside parameter-matched comparisons.

The stream fork/join stays outside the single AOTAutograd graph because the
project's custom PyTorch build currently mishandles a meta-device node in a
cross-stream FlexAttention backward graph. The normal `torch.compile` training
mode remains supported through this explicit eager scheduling boundary.

## Paper-aligned data and optimization recipe

- Source: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  subset `sample-10BT`, train split.
- Tokenizer: [RoBERTa base](https://huggingface.co/FacebookAI/roberta-base),
  vocabulary 50,265, with tied input/output embeddings.
- Context: 1,024-token padding-free rows. `document_ids` prevent attention
  across packed document boundaries.
- Validation: none, matching the paper's infinite-data assumption.
- MLM: independently select 20% of non-special tokens; replace **every**
  selected token with `<mask>` and compute loss only at selected positions.
  This is not the original BERT 80/10/10 replacement mixture.
- Optimizer: AdamW, peak LR `6e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`,
  weight decay `0.1`, global gradient clipping `1.0`.
- Schedule: 500 linear-warmup steps, then cosine decay through step 620 to
  `0.1` of peak LR (`6e-5`).
- Precision/runtime reconstruction: BF16, TF32 enabled, FlexAttention,
  `torch.compile`, and PyTorch DDP.

The exact accounting is:

```text
global batch          = 2,048 sequences
positions per step    = 2,048 * 1,024 = 2,097,152
optimizer steps       = 620
training rows         = 620 * 2,048 = 1,269,760
scheduled positions   = 620 * 2,048 * 1,024 = 1,300,234,240
```

The paired two-node launcher assumes eight H100s (two nodes, four GPUs per
node). Its starting batch decomposition preserves the same global batch:

| Variant | Microbatch/GPU | World size | Accumulation | Global batch |
| --- | ---: | ---: | ---: | ---: |
| Real | 32 | 8 | 8 | 2,048 |
| Multispace | 8 | 8 | 32 | 2,048 |

The multispace microbatch remains conservative: a full 28-layer,
sequence-length-1,024 BF16 optimizer-step smoke at batch 8 peaked at 21.965 GiB
allocated on a 44.422 GiB NVIDIA A40. Any further increase must first pass the
same production-geometry GPU gate, while accumulation must be reduced so the
global batch remains 2,048.

### Current dual-Flex memory limitation

The dual-number Flex backend is hybrid. FlexAttention computes the primal
branch, but the exact dual tangent uses a dense JVP and materializes the
document mask at shape `B x 4 x S x S` for the four dual heads. Therefore the
dual branch does not yet inherit FlexAttention's block-sparse memory scaling.
At `S=1024`, this can dominate multispace peak memory and throughput. It does
not change the algebra or mask semantics, but it makes the real and
multispace runs parameter-matched rather than compute- or memory-matched.

## Explicit reconstruction choices

The paper does not specify enough state for bitwise checkpoint reproduction.
This repository makes the missing choices explicit:

- FineWeb-Edu revision
  `fc9850dff5e2d0f8f776efe41b24a1c49556cfc5` (the commit currently named by
  the `v1.0.0` branch); take source rows in order until approximately 1.6B
  source tokens, then tokenize and pack enough full rows for training. This
  immutable source commit and the explicit `dataset.training_schedule`
  metadata (`optimizer_steps: 620`, `global_sequences: 2048`,
  `required_token_positions: 1300234240`) are reconstruction choices, not
  revisions or metadata reported by the paper.
- `FacebookAI/roberta-base` revision
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- Seed `0`, initialization range `0.02`, RMSNorm epsilon `1e-5`, zero dropout,
  tied embeddings, and no LM-head bias.
- AdamW epsilon `1e-8`, BF16/TF32, FlexAttention, compilation, ordinary DDP,
  and the two-node/four-H100-per-node topology above.
- Deterministic preprocessing logic and a recorded manifest, but not the
  authors' unreleased exact document order, random streams, or checkpoint.

These choices should be recorded with results rather than attributed to the
paper where its description is silent.

## Read-only validation and dry run

From the NeoBERT repository root, with the prepared environment selected:

```bash
cd /mnt/nfs/home/st171793/ComplexAttention/NeoBERT

export OPTIBERT_PYTHON=/shared/path/optibertneo-h100/bin/python
export OPTIBERT_DATASET=/shared/path/datasets/fineweb_edu_roberta_1p6b

"$OPTIBERT_PYTHON" scripts/pretraining/preflight_optibertneo.py --config-only
"$OPTIBERT_PYTHON" scripts/pretraining/validate_optibertneo_pair.py

for variant in real multispace; do
  DRY_RUN=1 \
  NUM_MACHINES=2 \
  GPUS_PER_NODE=4 \
  EXPECTED_NUM_MACHINES=2 \
  EXPECTED_WORLD_SIZE=8 \
  OPTIBERT_DATASET="$OPTIBERT_DATASET" \
  PYTHON_BIN="$OPTIBERT_PYTHON" \
    bash jobs/optibertneo-1p3b.sh "$variant"
done
```

The real dry run must print `micro_batch=32 gradient_accumulation=8`; the
multispace dry run must print `micro_batch=8 gradient_accumulation=32`. Both
must print `global_batch=2048`, `steps=620`, and
`scheduled_tokens=1300234240`. These commands do not prepare data or start
training.

The disposable GPU gate is a separate, test-only one-A100 job:

```bash
sbatch jobs/slurm/test-optibertneo-pair-a100.sbatch
```

It runs all paired static contracts inside the GPU allocation, then performs
CUDA BF16 forward/backward/AdamW checks for both variants. It compiles a
production-width one-layer model at sequence length 1,024 and exercises the
full 28-layer graph at a conservative sequence length of 64. It never loads
FineWeb-Edu, invokes `pretrain.py`, writes checkpoints, or submits another job.

Validation job `67877` completed successfully on an NVIDIA A40 (SM86) with
CUDA 12.6: 56 tests and 32 subtests passed, followed by compiled one-layer
1,024-token and full-depth 64-token BF16 optimizer steps for both variants.
It also completed full 28-layer, 1,024-token, compiled BF16 optimizer steps:
the real model at batch 1 peaked at 3.886 GiB allocated, and the multispace
model at batch 8 peaked at 21.965 GiB allocated (22.549 GiB reserved). Both
had finite loss and gradients and performed an AdamW update. The measured
elapsed times, including construction and compilation, were 53.473 s and
235.014 s respectively; they are smoke timings, not steady-state throughput.
After deadline, submission, and resume hardening, follow-up validation job
`67951` passed the expanded 72-test/55-subtest suite and repeated all default
CUDA BF16 optimizer-step checks successfully. Neither job read training data
or entered the pretraining pipeline.
The A40 was selected only because all A100s were occupied; the checked-in job
still defaults to A100 and should be rerun there before a production campaign.

## Future cluster execution (not submitted)

The intended gate order is data preparation, dataset preflight, distributed
NCCL smoke, short disposable GPU training smoke for each variant, submission
helper dry run, and only then the two full runs. Supply site-specific Slurm
account, partition, QoS, reservation, and time limits on the command line.

Prepare data once on a CPU/data-transfer partition with outbound access:

```bash
sbatch \
  --partition=<cpu-or-data-partition> \
  --account=<slurm-account> \
  --export=ALL \
  jobs/slurm/prepare-optibertneo-data.sbatch

"$OPTIBERT_PYTHON" scripts/pretraining/preflight_optibertneo.py \
  --dataset "$OPTIBERT_DATASET"
```

Validate the two-node/eight-GPU communication path:

```bash
sbatch \
  --partition=<h100-partition> \
  --account=<slurm-account> \
  --export=ALL \
  jobs/slurm/optibertneo-nccl-smoke-2n8g.sbatch
```

After the dataset and environment exist, print the exact two full submission
commands without submitting either job. Dry-run mode is the helper's default,
so omitting `DRY_RUN` is safe:

```bash
H100_PARTITION=<h100-partition> \
H100_ACCOUNT=<slurm-account> \
OPTIBERT_PYTHON="$OPTIBERT_PYTHON" \
OPTIBERT_DATASET="$OPTIBERT_DATASET" \
RUNS_ROOT=/shared/path/runs/optibertneo-paired-1p3b \
  bash jobs/submit-optibertneo-pair.sh both
```

Only after both variants pass their GPU smoke and the printed commands are
reviewed should a user deliberately disable dry-run mode and supply the second,
explicit affirmative gate:

```bash
DRY_RUN=0 \
CONFIRM_FULL_SUBMISSION=YES \
H100_PARTITION=<h100-partition> \
H100_ACCOUNT=<slurm-account> \
OPTIBERT_PYTHON="$OPTIBERT_PYTHON" \
OPTIBERT_DATASET="$OPTIBERT_DATASET" \
RUNS_ROOT=/shared/path/runs/optibertneo-paired-1p3b \
  bash jobs/submit-optibertneo-pair.sh both
```

The helper rejects inherited `SMOKE_TEST`, `GLOBAL_SEQUENCES`, `MICRO_BATCH`,
`NUM_MACHINES`, `GPUS_PER_NODE`, `MACHINE_RANK`, `EXPECTED_NUM_MACHINES`,
`EXPECTED_WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`, and `ACCELERATE_CONFIG`
values. Unset any of those variables before invoking it. Every printed or
submitted job explicitly uses full-run mode, a global batch of 2,048, the
two-node/eight-GPU DDP topology, and the validated per-GPU microbatch (`32` for
real and `8` for multispace). The allocation derives its own master address,
port, and per-node machine rank from Slurm.

That last command is intentionally documented but was **not run** while this
pipeline was prepared. The helper creates independent stable `real/` and
`multispace/` run roots so their resumable checkpoints cannot collide.

The supplied paired Slurm template has a four-hour wall clock and defaults
`MAX_TIME_SECONDS=13200` (3h40m). The job converts that interval to one absolute
coordinated-stop deadline before dataset/runtime preflight, preserving the
remaining 20 minutes for checkpoint I/O. At each optimizer boundary, the
trainer also reserves the duration of the preceding accumulation cycle and
will not begin another cycle without that runway. Before the first measured
cycle, a conservative 1,200-second runway floor applies. A persistent
`resume_signature.json` binds each run root to its dataset manifest and Arrow
fingerprint, topology, microbatch/accumulation, seed, model, tokenizer,
masking/data-loader semantics, optimizer, scheduler, and trainer semantics;
missing or mismatched identity is rejected before model construction. If a run
does not reach step 620 in one allocation, all ranks write a complete
checkpoint and a later submission against the same variant run root resumes
it. A time-limited partial run is not exported as `final_model`; export happens
only after all 620 optimizer steps complete.

## Relevant files

- Real model: `conf/model/optibertneo-198m.yaml`
- Multispace model: `conf/model/optibertneo-198m-multispace.yaml`
- Paired parameter validator: `scripts/pretraining/validate_optibertneo_pair.py`
- A100 optimizer-step smoke: `scripts/pretraining/smoke_optibertneo_pair.py`
- FineWeb-Edu recipe: `conf/dataset/fineweb_edu.yaml`
- MLM recipe: `conf/datacollator/mlm_20.yaml`
- Per-node launcher: `jobs/optibertneo-1p3b.sh`
- Two-node job: `jobs/slurm/optibertneo-paired-1p3b-2n8g.sbatch`
- Test-only A100 job: `jobs/slurm/test-optibertneo-pair-a100.sbatch`
- Resume/checkpoint contracts: `tests/test_optibertneo_checkpointing.py`
- Explicit submit/dry-run helper: `jobs/submit-optibertneo-pair.sh`
