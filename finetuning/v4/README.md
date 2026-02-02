# Entity Injection v4: 3-Stage Training with Per-Token Gating

## Overview

Version 4 introduces **3-stage curriculum training** and **per-token gating** to address the gate collapse problem observed in v3. The gate is frozen during early training to ensure the entity classifier learns meaningful representations before the gate can suppress them.

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
│  │                 │ (type_hidden)         │      │         │
│  │                 ├───────────────────┐   │      │         │
│  │                 │                   │   │      │         │
│  │                 ▼                   ▼   │      │         │
│  │   ┌─────────────────────┐ ┌─────────────────┐  │         │
│  │   │  entity_delta_proj  │ │   gate_proj     │  │         │
│  │   │   Linear(D → D)     │ │ Linear(D → 1)   │  │         │
│  │   └──────────┬──────────┘ └────────┬────────┘  │         │
│  │              │                     │           │         │
│  │              │                     ▼           │         │
│  │              │            ┌─────────────────┐  │         │
│  │              │            │    sigmoid()    │  │         │
│  │              │            │  Per-Token Gate │  │         │
│  │              │            │    [B, T, 1]    │  │         │
│  │              │            └────────┬────────┘  │         │
│  │              │                     │           │         │
│  │              │ type_delta          │ gate      │         │
│  │              └──────────┬──────────┘           │         │
│  │                         │                      │         │
│  │                         ▼                      │         │
│  │   ┌─────────────────────────────────────────┐  │         │
│  │   │      type_enriched = text + gate*delta  │  │         │
│  │   └─────────────────────────────────────────┘  │         │
│  │                                                │         │
│  │                  type_head ───► Entity Loss    │         │
│  │                                                │         │
│  └────────────────────────────────────────────────┘         │
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
| `gate_proj` | **Per-token gate network** Linear(D → 1) + sigmoid |

### Injection Formula

```python
type_hidden = entity_encoder(text_embeddings)
type_delta = entity_delta_proj(type_hidden)
learned_gate = sigmoid(gate_proj(type_hidden))  # [B, T, 1]

# Gate value depends on training stage:
# Stage 1: gate = 0 (no injection)
# Stage 2a: gate = fixed_gate_value
# Stage 2b: gate = clamp(learned_gate, min=gate_min)

type_enriched = text_embeddings + gate * type_delta
```

## Training Scheme

### 3-Stage Curriculum Training

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  STAGE 1                   STAGE 2a                  STAGE 2b        │
│  Classifier Warmup         Fixed Gate               Learned Gate     │
│  (0 → 16k steps)           (16k → 32k steps)        (32k+ steps)     │
│                                                                      │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│  │                 │      │                 │      │                 │
│  │  Base: FROZEN   │      │  Base: ACTIVE   │      │  Base: ACTIVE   │
│  │  Classifier:    │  →   │  Classifier:    │  →   │  Classifier:    │
│  │    TRAINING     │      │    TRAINING     │      │    TRAINING     │
│  │  Gate: 0        │      │  Gate: 0.1      │      │  Gate: Learned  │
│  │                 │      │  (fixed)        │      │  (min=0.05)     │
│  │                 │      │                 │      │                 │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘
│                                                                      │
│  Goal: Classifier         Goal: Stable             Goal: Adaptive    │
│  learns entity            integration of           per-token         │
│  representations          entity info              gating            │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Stage Details

| Stage | Steps | Base Model | Entity Classifier | Gate Value |
|-------|-------|------------|-------------------|------------|
| **1** | 0 → 16k | Frozen | Training | 0 (no injection) |
| **2a** | 16k → 32k | Training | Training | 0.1 (fixed) |
| **2b** | 32k+ | Training | Training | Learned (min 0.05) |

### Loss Function

```
total_loss = tts_loss + entity_loss_weight * entity_loss
```

- `entity_loss_weight`: 0.5 (increased from v3)
- Special tokens (padding, BOS, EOS) are masked with label=-100

### Optimizer Configuration

| Parameter Group | Learning Rate | Notes |
|----------------|---------------|-------|
| Base model | 1e-5 | Frozen during Stage 1 |
| Entity module | 1e-5 × 10 = 1e-4 | Higher LR for faster learning |

## Configuration

### Default Hyperparameters

```bash
CLASSIFIER_WARMUP_STEPS=16000   # End of Stage 1
GATE_FREEZE_STEPS=32000         # End of Stage 2a  
FIXED_GATE_VALUE=0.1            # Gate during Stage 2a
GATE_MIN=0.05                   # Minimum gate during Stage 2b
ENTITY_LOSS_WEIGHT=0.5          # Increased from 0.1
```

## Known Issues

1. **Gate Collapse Persistence**: Even with 3-stage training, gate can still collapse after Stage 2b begins
2. **Entity Loss Fluctuation**: Loss still fluctuates significantly, suggesting classification instability
3. **Training from Scratch**: Entity encoder (TransformerEncoder) must learn from random initialization

## Usage

```bash
./run_finetune_v4.sh \
    --batch_size 16 \
    --lr 1e-5 \
    --epochs 20 \
    --entity_loss_weight 0.5 \
    --classifier_warmup_steps 16000 \
    --gate_freeze_steps 32000 \
    --fixed_gate_value 0.1 \
    --gate_min 0.05
```

## Files

| File | Description |
|------|-------------|
| `sft_12hz_with_EntityInjection_v4.py` | Main training script with 3-stage logic |
| `dataset.py` | Dataset class with entity handling |
| `run_finetune_v4.sh` | Training launcher script |
