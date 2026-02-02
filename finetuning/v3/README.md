# Entity Injection v3: Gate-Based Residual Injection

## Overview

Version 3 introduces a **learnable global gate** for entity injection. A TransformerEncoder processes text embeddings to predict entity types, and the entity information is injected back into the embeddings via a gated residual connection.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Base Qwen3-TTS Model                   │
│  ┌───────────────────┐                                      │
│  │  Text Embeddings  │ ───────────────────────────┐         │
│  │      [B, T, D]    │                            │         │
│  └─────────┬─────────┘                            │         │
│            │                                      │         │
│            ▼                                      │         │
│  ┌─────────────────────────────────────────┐      │         │
│  │       EntityInjectionModule             │      │         │
│  │                                         │      │         │
│  │   ┌───────────────────────────────┐     │      │         │
│  │   │ TransformerEncoder (2 layers) │     │      │         │
│  │   │    d_model=1024, heads=8      │     │      │         │
│  │   └─────────────┬─────────────────┘     │      │         │
│  │                 │                       │      │         │
│  │                 ▼                       │      │         │
│  │   ┌───────────────────────────────┐     │      │         │
│  │   │    entity_delta_proj          │     │      │         │
│  │   │    Linear(D → D)              │──┐  │      │         │
│  │   └───────────────────────────────┘  │  │      │         │
│  │                 │                    │  │      │         │
│  │                 ▼                    │  │      │         │
│  │   ┌───────────────────────────────┐  │  │      │         │
│  │   │     type_head (classifier)    │  │  │      │         │
│  │   │   Linear(D → num_entities)    │  │  │      │         │
│  │   └─────────────┬─────────────────┘  │  │      │         │
│  │                 │                    │  │      │         │
│  │                 ▼                    │  │      │         │
│  │        Entity Loss (CE)              │  │      │         │
│  │                                      │  │      │         │
│  │   gate (learnable scalar, init=0.1) ─┤  │      │         │
│  │                                      │  │      │         │
│  └──────────────────────────────────────┼──┼──────┘         │
│                                         │  │                │
│            ┌────────────────────────────┘  │                │
│            │                               │                │
│            ▼                               ▼                │
│   ┌───────────────────────────────────────────────┐         │
│   │  type_enriched = text + gate * delta          │         │
│   └─────────────────────┬─────────────────────────┘         │
│                         │                                   │
│                         ▼                                   │
│               text_projection → ...                         │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### EntityInjectionModule

| Component | Description |
|-----------|-------------|
| `entity_encoder` | 2-layer TransformerEncoder (d=1024, 8 heads) |
| `entity_delta_proj` | Linear projection for residual delta |
| `type_head` | Entity classification head |
| `gate` | **Global learnable scalar** (initialized to 0.1) |

### Injection Formula

```python
type_delta = entity_delta_proj(entity_encoder(text_embeddings))
type_enriched = text_embeddings + gate * type_delta
```

## Training Scheme

### Single-Stage Training
- All components train together from the start
- No staged warmup or freezing
- Entity loss and TTS loss combined

### Loss Function

```
total_loss = tts_loss + entity_loss_weight * entity_loss
```

Where:
- `tts_loss`: Cross-entropy on codec prediction
- `entity_loss`: Cross-entropy on entity type prediction
- `entity_loss_weight`: 0.1 (default)

### Optimizer Configuration

| Parameter Group | Learning Rate |
|----------------|---------------|
| Base model | 1e-5 |
| Entity module | 1e-5 |

## Known Issues

1. **Gate Collapse**: The global gate tends to stay near its initialization (0.1) and doesn't learn effectively
2. **Loss Fluctuation**: Entity prediction loss fluctuates significantly (0.5 → 0.004) without converging
3. **Limited Context**: Single scalar gate cannot capture per-token or per-entity variations

## Usage

```bash
./run_finetune_v3.sh \
    --batch_size 16 \
    --lr 1e-5 \
    --epochs 20 \
    --entity_loss_weight 0.1
```

## Files

| File | Description |
|------|-------------|
| `sft_12hz_with_EntityInjection_v3.py` | Main training script |
| `sft_12hz_with_EntityInjection_v3_infer.py` | Inference script |
| `dataset.py` | Dataset class with entity handling |
| `run_finetune_v3.sh` | Training launcher script |
