# Paired SuperGLUE evaluation

This suite compares the two exact 99,985,152-parameter FineWeb-Edu
checkpoints under an identical three-seed supervised protocol.  It reports
official validation metrics because the main-task test labels require a
SuperGLUE submission.  These are external, non-leaderboard results.

## Immutable inputs

- Dataset mirror: `aps/super_glue`
- Dataset revision: `3de24cf8022e94f4ee4b9d55a6f539891524d646`
- Local data: `.cache/evaluations/data/superglue-3de24cf8022e`
- Per-file SHA-256 and row counts:
  `scripts/fineweb_evaluation/superglue_data_manifest.json`
- Metric reference: Hugging Face Datasets tag `2.14.6`, commit
  `06c3ffb8d068b6307b247164b10f7c7311cefed4`.  The manifest recorded by each
  run pins hashes for both `super_glue.py` and `record_evaluation.py`.
- Classifier implementation: BabyLM evaluator commit
  `6f825c291e2c4c78ad33b1935fd64d45f52642dc`.

## Tasks and scores

| Task | Validation metric | Primary task score |
|---|---|---|
| BoolQ | accuracy | accuracy |
| CB | accuracy, macro-F1 | mean of accuracy and F1 |
| COPA | accuracy | accuracy |
| MultiRC | F1a, per-question macro-F1 (F1m), exact match | mean of F1a and exact match |
| ReCoRD | normalized token F1, exact match | mean of F1 and exact match |
| RTE | accuracy | accuracy |
| WiC | accuracy | accuracy |
| WSC-fixed | accuracy | accuracy |

The overall SuperGLUE score is the unweighted mean of the eight primary task
scores.  AX-b is reported with Matthews correlation and AX-g with accuracy.
Both use the trained RTE head as transfer diagnostics and are excluded from
the overall score.

ReCoRD uses an extractive start/end head.  Training windows are centered on a
gold answer occurrence; validation predictions score every supplied entity
span and select the highest-scoring entity.  This preserves the official
entity-constrained answer space and F1/exact-match calculation.

## Controlled protocol

- Seeds: 42, 43, 44 for each model and task.
- Optimizer: AdamW, learning rate `3e-5`, weight decay `0.01`.
- Scheduler: cosine, 6% warmup; gradient clipping at 1.0.
- Sequence length: 512; left padding and final non-padding-token pooling.
- Precision: FP32 with TF32 disabled and deterministic algorithms enabled.
- Epochs: BoolQ 5, CB 20, COPA 20, MultiRC 5, ReCoRD 2, RTE 10, WiC 10,
  WSC 20.
- Batch size: 16 (ReCoRD 8); evaluation batch size 64 (ReCoRD 32).
- Every preflight, training, evaluation, fairness gate, and aggregation job is
  allocated one A100 GPU.

The exported checkpoints retain their learned tensors.  For padded supervised
batches, both variants select the algebraically equivalent PyTorch attention
backend through `scripts/fineweb_eval/babylm_finetune_compat/sitecustomize.py`.
Checkpoint attention remains Flash.  The compatibility layer isolates encoder
construction RNG, and the gate requires identical classifier-head SHA-256 for
both variants for every task and seed.

## Job chain and outputs

Submit from the NeoBERT root:

```bash
bash jobs/fineweb_evaluation/submit_superglue.sh
```

The chain is `paired preflight -> fairness gate -> 48 full runs -> aggregate`.
Results are written under `logs/fineweb_evaluation/superglue`:

- `preflight/fairness_gate.json`
- `full/<task>/<variant>/seed-<seed>/report.json`
- `full/<task>/<variant>/seed-<seed>/predictions.jsonl`
- `aggregate/summary.json`
- `aggregate/summary.md`
