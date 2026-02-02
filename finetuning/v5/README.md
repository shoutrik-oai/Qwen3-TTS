# Entity Injection v5: FiLM Conditioning with Pretrained Encoder

## Overview

Version 5 replaces the gate-based residual injection with **Feature-wise Linear Modulation (FiLM)** and uses a **pretrained BERT encoder** instead of training a TransformerEncoder from scratch. This addresses the gate collapse problem and enables faster convergence through transfer learning.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Base Qwen3-TTS Model                            │
│  ┌───────────────────┐                                                  │
│  │  Text Embeddings  │                                                  │
│  │  [B, T, D_qwen]   │                                                  │
│  │     (1024)        │                                                  │
│  └─────────┬─────────┘                                                  │
│            │                                                            │
│            ▼                                                            │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                    EntityInjectionModule                           │ │
│  │                                                                    │ │
│  │   ┌──────────────────────────────────────────────────────────────┐ │ │
│  │   │                   PRETRAINED ENCODER                         │ │ │
│  │   │                                                              │ │ │
│  │   │   ┌────────────────┐   ┌─────────────────────┐              │ │ │
│  │   │   │  input_proj    │   │   Pretrained BERT   │              │ │ │
│  │   │   │ (D_qwen→D_bert)│ → │  (DistilBERT/BERT)  │              │ │ │
│  │   │   │   1024 → 768   │   │   Finetuned 0.1x LR │              │ │ │
│  │   │   └────────────────┘   └──────────┬──────────┘              │ │ │
│  │   │                                   │                         │ │ │
│  │   │                                   ▼                         │ │ │
│  │   │                       ┌────────────────────┐                │ │ │
│  │   │                       │    output_proj     │                │ │ │
│  │   │                       │ (D_bert→D_qwen)    │                │ │ │
│  │   │                       │   768 → 1024       │                │ │ │
│  │   │                       └─────────┬──────────┘                │ │ │
│  │   │                                 │                           │ │ │
│  │   └─────────────────────────────────┼───────────────────────────┘ │ │
│  │                                     │                             │ │
│  │                                     ▼                             │ │
│  │                       ┌─────────────────────────┐                 │ │
│  │                       │      entity_head        │                 │ │
│  │                       │  Linear(D → num_types)  │                 │ │
│  │                       └────────────┬────────────┘                 │ │
│  │                                    │                              │ │
│  │                     entity_logits  │                              │ │
│  │                                    ▼                              │ │
│  │   ┌────────────────────────────────────────────────────────────┐  │ │
│  │   │                      FiLM Layer                            │  │ │
│  │   │                                                            │  │ │
│  │   │   type_probs = softmax(entity_logits / τ)                  │  │ │
│  │   │   type_emb = type_probs @ entity_type_embeddings           │  │ │
│  │   │                                                            │  │ │
│  │   │   ┌─────────────────────────────────────────────┐          │  │ │
│  │   │   │  film_mlp: Linear → GELU → Linear           │          │  │ │
│  │   │   │           (D → 2D → 2D)                     │          │  │ │
│  │   │   └─────────────────┬───────────────────────────┘          │  │ │
│  │   │                     │                                      │  │ │
│  │   │                     ▼                                      │  │ │
│  │   │   ┌─────────────────────────────────────────────┐          │  │ │
│  │   │   │  [gamma_offset, beta] = chunk(output, 2)    │          │  │ │
│  │   │   │  gamma = 1.0 + 0.5 * tanh(gamma_offset)     │          │  │ │
│  │   │   │  → gamma ∈ [0.5, 1.5]                       │          │  │ │
│  │   │   └─────────────────┬───────────────────────────┘          │  │ │
│  │   │                     │                                      │  │ │
│  │   │                     ▼                                      │  │ │
│  │   │   ┌─────────────────────────────────────────────┐          │  │ │
│  │   │   │  conditioned = gamma * text_emb + beta      │          │  │ │
│  │   │   │  (FiLM: Feature-wise Linear Modulation)     │          │  │ │
│  │   │   └─────────────────────────────────────────────┘          │  │ │
│  │   │                                                            │  │ │
│  │   └────────────────────────────────────────────────────────────┘  │ │
│  │                                                                   │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                     │                                  │
│                                     ▼                                  │
│                          text_projection → ...                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Key Components

### EntityInjectionModule

| Component | Description |
|-----------|-------------|
| `pretrained_encoder` | DistilBERT/BERT (finetuned with 0.1x LR) |
| `input_proj` | Linear(1024 → 768) - Qwen to BERT dimension |
| `output_proj` | Linear(768 → 1024) - BERT to Qwen dimension |
| `entity_head` | Entity classification head (trained from scratch) |
| `film_layer` | FiLM conditioning layer (trained from scratch) |

### FiLM Layer

| Component | Description |
|-----------|-------------|
| `entity_type_embeddings` | Learnable embeddings for each entity type |
| `film_mlp` | MLP to generate γ and β from entity embeddings |

### FiLM vs Gate

| Approach | Formula | Failure Mode |
|----------|---------|--------------|
| **Gate (v3/v4)** | `H + g × Δ` | g → 0 (collapses to identity) |
| **FiLM (v5)** | `γ × H + β` | γ ∈ [0.5, 1.5], always modifies H |

## Training Scheme

### Simplified Single-Stage Training

Unlike v4's 3-stage curriculum, v5 uses **single-stage training** because:
1. FiLM cannot collapse (γ is bounded away from 0)
2. Pretrained encoder provides good initial representations
3. Differential learning rates handle adaptation naturally

### Learning Rate Strategy

| Component | LR Multiplier | Purpose |
|-----------|---------------|---------|
| Base TTS model | 1× | Standard finetuning |
| Pretrained encoder (BERT) | 0.1× | Gentle finetuning |
| Projection layers | 10× | Train from scratch |
| Entity head | 10× | Train from scratch |
| FiLM layer | 10× | Train from scratch |

### Loss Function

```python
total_loss = tts_loss + entity_loss_weight * entity_loss

# DDP compatibility - ensure all params in graph
entity_loss += 0.0 * gamma.sum() + 0.0 * beta.sum()
```

## Pretrained Encoder Options

| Model | Hidden Size | Params | Speed |
|-------|-------------|--------|-------|
| `prajjwal1/bert-tiny` | 128 | 4.4M | Fastest |
| `prajjwal1/bert-mini` | 256 | 11M | Fast |
| `distilbert-base-uncased` | 768 | 66M | Medium (default) |
| `bert-base-uncased` | 768 | 110M | Slower |
| `none` | N/A | N/A | Train from scratch |

## Configuration

### Default Hyperparameters

```bash
PRETRAINED_ENCODER="distilbert-base-uncased"
FREEZE_ENCODER=false
ENTITY_LOSS_WEIGHT=0.1
LEARNING_RATE=1e-5
```

## Advantages over v3/v4

| Aspect | v3/v4 | v5 |
|--------|-------|-----|
| **Gate Collapse** | Common issue | Impossible (γ bounded) |
| **Encoder Init** | Random | Pretrained BERT |
| **Training Stages** | 3 stages (v4) | 1 stage |
| **Conditioning** | Additive (H + g×Δ) | Affine (γ×H + β) |
| **Convergence** | Slow | Faster (transfer learning) |

## Usage

### Standard Training (DistilBERT)
```bash
python sft_12hz_with_EntityInjection_v5.py \
    --pretrained_encoder distilbert-base-uncased \
    --lr 1e-5 \
    --entity_loss_weight 0.1
```

### Tiny BERT (Faster)
```bash
python sft_12hz_with_EntityInjection_v5.py \
    --pretrained_encoder prajjwal1/bert-tiny
```

### Freeze BERT (Only Train FiLM + Projections)
```bash
python sft_12hz_with_EntityInjection_v5.py \
    --pretrained_encoder distilbert-base-uncased \
    --freeze_encoder
```

### Train from Scratch (No Pretrained)
```bash
python sft_12hz_with_EntityInjection_v5.py \
    --pretrained_encoder none
```

## Files

| File | Description |
|------|-------------|
| `sft_12hz_with_EntityInjection_v5.py` | Main training script with FiLM + BERT |
| `sft_12hz_with_EntityInjection_v3_infer.py` | Inference script (shared with v3) |
| `dataset.py` | Dataset class with entity handling |
| `run_finetune_v3.sh` | Training launcher script |

## Mathematical Details

### FiLM Conditioning

```
Given:
  - H: text hidden states [B, T, D]
  - entity_logits: [B, T, num_entities]
  - E: entity type embeddings [num_entities, D]

Compute:
  1. type_probs = softmax(entity_logits / τ)     # τ = temperature
  2. type_emb = type_probs @ E                   # soft entity embedding
  3. [γ_offset, β] = MLP(type_emb)
  4. γ = 1 + 0.5 * tanh(γ_offset)                # γ ∈ [0.5, 1.5]
  5. H_conditioned = γ * H + β                   # affine transform
```

### Why FiLM Can't Collapse

- **Gate approach**: `H + g × Δ` can collapse when `g → 0`, reverting to identity
- **FiLM approach**: `γ × H + β` where:
  - `γ ∈ [0.5, 1.5]` (bounded by tanh)
  - Even at initialization (γ ≈ 1, β ≈ 0), the path through entity embeddings and FiLM MLP is always active
  - Gradients always flow through entity_type_embeddings and film_mlp
