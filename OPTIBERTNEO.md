# OptiBERTneo 198M / 1.3B comparison

This experiment trains two encoder-only MLMs on exactly the same data order and token schedule:

| Variant | Layers | Width | Heads | Attention schedule | Non-embedding parameters |
| --- | ---: | ---: | ---: | --- | ---: |
| `baseline` | 28 | 768 | 12 x 64 | 28 real | about 198.23M |
| `mixed` | 28 | 640 | 10 x 64 | 10 complex, 9 dual, 9 split | about 198.31M |

The paper's 198M count is `12 * 768^2 * 28 = 198,180,864` and omits embeddings and small normalization terms. The mixed width and SwiGLU size are chosen so its non-embedding count remains within 0.1% of that budget. The embedding table is smaller in the mixed model because it is narrower.

## Shared pretraining recipe

- FineWeb-Edu, RoBERTa BPE tokenizer, vocabulary 50,265
- maximum sequence length 1,024
- document-aware sequence packing with no cross-document attention
- 20% MLM masking, with every selected token replaced by `<mask>`
- AdamW, peak LR `6e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`
- gradient clipping at 1.0, BF16, TF32
- 500 warmup steps, cosine decay to 10% of peak LR at step 620
- global batch 2,048 sequences = 2,097,152 tokens
- 620 optimizer steps = 1,300,234,240 scheduled tokens

The baseline follows the paper architecture and training settings. The paper states that OptiBERTneo retains NeoBERT's batch size and learning rate; those values are taken from the NeoBERT training appendix. PyTorch FlexAttention implements the paper's document-restricted packed attention for both variants.

The authors did not publish an OptiBERT training repository or exact data order with the paper. This setup is therefore a controlled public reconstruction of the reported recipe, not a bitwise reproduction of their checkpoint.

## Prepare data

Install the editable projects in the training environment first, then preprocess one shared dataset:

```bash
cd /path/to/ComplexAttention
pip install -e .
pip install -e NeoBERT --no-deps

cd NeoBERT
python scripts/pretraining/preprocess.py dataset=fineweb_edu tokenizer=roberta
```

The configuration selects about 1.6B source tokens from the public FineWeb-Edu `sample-10BT` subset. It chunks long documents and writes one deterministic set of prepacked 1,024-token rows, so both models see the same document groupings despite using different microbatch sizes. Segments are always bounded by RoBERTa BOS/EOS tokens; at most two padding positions are needed when a new segment cannot fit at the end of a row. This leaves enough RoBERTa tokens for the 1.3B-token schedule without tokenizing all 10B tokens.

## Validate model sizes

```bash
cd /path/to/ComplexAttention/NeoBERT
python scripts/pretraining/inspect_optibertneo.py all --check
```

## Launch training

First run two optimizer steps through each model on the target GPU software stack:

```bash
SMOKE_TEST=1 WANDB_MODE=disabled bash jobs/optibertneo-1p3b.sh baseline
SMOKE_TEST=1 WANDB_MODE=disabled bash jobs/optibertneo-1p3b.sh mixed
```

Run the launcher once per node inside the allocation. On a single 8-GPU node:

```bash
cd /path/to/ComplexAttention/NeoBERT
bash jobs/optibertneo-1p3b.sh baseline
bash jobs/optibertneo-1p3b.sh mixed
```

For Slurm multi-node jobs, launch one copy per node:

```bash
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks-per-node=1 \
  bash jobs/optibertneo-1p3b.sh baseline
```

Useful environment overrides are `GPUS_PER_NODE`, `MICRO_BATCH`, `OPTIBERT_DATASET`, `RUN_ROOT`, `WANDB_MODE`, `MASTER_ADDR`, and `MASTER_PORT`. Defaults on 8 GPUs are microbatch 32 and accumulation 8 for the baseline, and microbatch 4 and accumulation 64 for the mixed model. Reduce `MICRO_BATCH` if the dual blocks exceed memory; the launcher increases accumulation automatically to preserve the global batch.

## Compare results

Report MLM loss against tokens, wall-clock time, tokens/second, peak GPU memory, and downstream GLUE/MTEB results. Parameter matching does not make compute identical: split attention invokes two efficient attention kernels, while dual attention also computes the exact tangent/JVP. Include both equal-token quality and measured training cost in the final comparison.
