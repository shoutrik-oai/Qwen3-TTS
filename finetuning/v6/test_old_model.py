#!/usr/bin/env python3
"""
Diagnostic script to test the old model and check for:
1. Unbounded tensors
2. Masking correctness
3. Numerical issues
"""

import torch
import json
import os

from safetensors.torch import load_file
from transformers import AutoConfig
from qwen_tts import Qwen3TTSModel

# Import the OLD EntityInjectionModule (without fixes)
# We'll manually create a version without the beta bounding
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class FiLMLayer(nn.Module):
    def __init__(self, hidden_size, num_entities):
        super().__init__()
        self.entity_type_embeddings = nn.Embedding(num_entities, hidden_size)
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.entity_type_embeddings.weight, std=0.02)
        final_layer = self.film_mlp[-1]
        nn.init.normal_(final_layer.weight, std=0.001)
        nn.init.zeros_(final_layer.bias)
    
    def forward(self, hidden_states, entity_logits, temperature=1.0):
        type_probs = F.softmax(entity_logits / temperature, dim=-1)
        type_emb = type_probs @ self.entity_type_embeddings.weight
        
        film_params = self.film_mlp(type_emb)
        gamma_offset, beta = film_params.chunk(2, dim=-1)
        
        # Original: beta is unbounded!
        gamma = 1.0 + 0.5 * torch.tanh(gamma_offset)
        
        conditioned = gamma * hidden_states + beta
        
        return conditioned, gamma, beta


class EntityInjectionModuleConfig:
    def __init__(self, hidden_size, num_entities, num_layers=2, num_heads=8, 
                 dim_feedforward=2048, dropout=0.1, entity_prob=0.0):
        self.hidden_size = hidden_size
        self.num_entities = num_entities
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.entity_prob = entity_prob


class EntityInjectionModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        qwen_hidden_size = config.hidden_size
        num_entities = config.num_entities
        self.num_entities = num_entities

        self.positional_embedding = SinusoidalPositionalEmbedding(config.hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=qwen_hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True
        )
        self.entity_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        encoder_output_dim = qwen_hidden_size
        
        self.entity_head = nn.Linear(encoder_output_dim, num_entities)
        self.entity_detector = nn.Linear(encoder_output_dim, 1)
        self.film_layer = FiLMLayer(qwen_hidden_size, num_entities)
        self.config = config
    
    def forward(self, text_embeddings, text_mask=None, return_all=False):
        """Forward with detailed outputs for debugging."""
        if text_mask is not None:
            src_key_padding_mask = (text_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Store original for comparison
        original_embeddings = text_embeddings.clone()
        
        # Add positional embeddings
        text_embeddings_with_pe = self.positional_embedding(text_embeddings)
        
        # Encoder
        entity_hidden = self.entity_encoder(text_embeddings_with_pe, src_key_padding_mask=src_key_padding_mask)
        
        # Predictions
        entity_logits = self.entity_head(entity_hidden)
        entity_detection_logits = self.entity_detector(entity_hidden)
        is_entity = torch.sigmoid(entity_detection_logits)
        
        # FiLM
        conditioned_film, gamma, beta = self.film_layer(text_embeddings_with_pe, entity_logits)
        
        # Blend
        conditioned = text_embeddings_with_pe * (1 - is_entity) + conditioned_film * is_entity
        
        if return_all:
            return {
                'original_embeddings': original_embeddings,
                'embeddings_with_pe': text_embeddings_with_pe,
                'entity_hidden': entity_hidden,
                'entity_logits': entity_logits,
                'entity_detection_logits': entity_detection_logits,
                'is_entity': is_entity,
                'gamma': gamma,
                'beta': beta,
                'conditioned_film': conditioned_film,
                'conditioned': conditioned,
                'src_key_padding_mask': src_key_padding_mask,
            }
        
        return conditioned, entity_logits, entity_detection_logits


def analyze_tensor(name, tensor, mask=None):
    """Analyze a tensor and print statistics."""
    print(f"\n{'='*60}")
    print(f"Tensor: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
    print(f"  Mean: {tensor.mean().item():.4f}")
    print(f"  Std: {tensor.std().item():.4f}")
    print(f"  Has NaN: {tensor.isnan().any().item()}")
    print(f"  Has Inf: {tensor.isinf().any().item()}")
    
    if mask is not None:
        masked = tensor[mask]
        unmasked = tensor[~mask]
        if masked.numel() > 0:
            print(f"  [Masked (text) positions]:")
            print(f"    Range: [{masked.min().item():.4f}, {masked.max().item():.4f}]")
            print(f"    Mean: {masked.mean().item():.4f}")
        if unmasked.numel() > 0:
            print(f"  [Unmasked (header/special) positions]:")
            print(f"    Range: [{unmasked.min().item():.4f}, {unmasked.max().item():.4f}]")
            print(f"    Mean: {unmasked.mean().item():.4f}")


def main():
    model_path = "/speech/arjun/shoutrik/Qwen3-TTS/finetuning/v6/experiments/entity_injection_v6.2/checkpoint-epoch-17"
    device = "cuda"
    
    print("="*80)
    print("DIAGNOSTIC TEST: Old Model Analysis")
    print("="*80)
    
    # Load Qwen3TTS
    print("\n1. Loading Qwen3TTS model...")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    
    # Load entity mapping
    print("\n2. Loading entity mapping...")
    entity_mapping_path = os.path.join(model_path, "entity_type_mapping.json")
    with open(entity_mapping_path, 'r') as f:
        entity_mapping = json.load(f)
    entity_config_dict = entity_mapping["entity_injection_config"]
    
    # Create entity injection module
    print("\n3. Creating EntityInjectionModule...")
    entity_injection_config = EntityInjectionModuleConfig(
        hidden_size=entity_config_dict["hidden_size"],
        num_entities=entity_config_dict["num_entities"],
        num_layers=entity_config_dict.get("num_layers", 2),
        num_heads=entity_config_dict.get("num_heads", 8),
        dim_feedforward=entity_config_dict.get("dim_feedforward", 2048),
        dropout=entity_config_dict.get("dropout", 0.1),
    )
    entity_injection_module = EntityInjectionModule(entity_injection_config)
    
    # Load weights
    entity_module_path = os.path.join(model_path, "entity_injection_module.safetensors")
    entity_state_dict = load_file(entity_module_path)
    entity_injection_module.load_state_dict(entity_state_dict)
    
    # Move to device
    model_dtype = next(qwen3tts.model.parameters()).dtype
    entity_injection_module = entity_injection_module.to(device).to(model_dtype)
    entity_injection_module.eval()
    
    print(f"  Model dtype: {model_dtype}")
    
    # Test text
    text = "Leave the file with Dr. Brown who lives on Pine dr."
    print(f"\n4. Test text: \"{text}\"")
    
    # Build input_ids like in inference
    config = AutoConfig.from_pretrained(model_path)
    assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    full_input_ids = qwen3tts.processor.tokenizer(assistant_text, return_tensors="pt").input_ids.to(device)
    full_input_ids = full_input_ids[:, :-5]
    text_ids_len = full_input_ids.shape[1]
    
    input_ids = torch.zeros((1, text_ids_len + 6), device=device, dtype=torch.long)
    input_ids[0, :3] = full_input_ids[0, :3]
    input_ids[0, 3:7] = config.tts_pad_token_id
    input_ids[0, 7] = config.tts_bos_token_id
    input_ids[0, 8:8 + text_ids_len - 3] = full_input_ids[0, 3:]
    input_ids[0, 8+text_ids_len-3] = config.tts_eos_token_id
    
    print(f"\n5. Input structure:")
    print(f"  Total tokens: {input_ids.shape[1]}")
    print(f"  Header (0-2): positions 0-2")
    print(f"  Pad (3-6): positions 3-6")
    print(f"  BOS (7): position 7")
    print(f"  Text (8-{8+text_ids_len-3-1}): positions 8-{8+text_ids_len-4}")
    print(f"  EOS ({8+text_ids_len-3}): position {8+text_ids_len-3}")
    
    # Create text_only_mask
    text_only_mask = torch.zeros((1, input_ids.shape[1]), dtype=torch.bool, device=device)
    text_only_mask[0, 8:8+text_ids_len-3] = True
    
    print(f"\n6. text_only_mask:")
    print(f"  True positions: 8 to {8+text_ids_len-4}")
    print(f"  Mask: {text_only_mask[0].tolist()}")
    
    # Get embeddings
    print("\n7. Getting text embeddings...")
    text_embed = qwen3tts.model.talker.model.text_embedding(input_ids)
    
    analyze_tensor("text_embed (input)", text_embed)
    
    # Run entity injection with detailed outputs
    print("\n8. Running entity injection...")
    with torch.no_grad():
        outputs = entity_injection_module(text_embed, text_mask=text_only_mask, return_all=True)
    
    # Expand mask for tensor analysis
    mask_expanded = text_only_mask.unsqueeze(-1).expand_as(text_embed)
    
    # Analyze all intermediate tensors
    analyze_tensor("positional_embedding.pe (buffer)", entity_injection_module.positional_embedding.pe[:, :input_ids.shape[1]])
    analyze_tensor("embeddings_with_pe", outputs['embeddings_with_pe'], mask_expanded)
    analyze_tensor("entity_hidden (encoder output)", outputs['entity_hidden'], mask_expanded)
    analyze_tensor("entity_logits", outputs['entity_logits'])
    analyze_tensor("entity_detection_logits", outputs['entity_detection_logits'])
    analyze_tensor("is_entity (sigmoid)", outputs['is_entity'])
    analyze_tensor("gamma", outputs['gamma'], mask_expanded)
    analyze_tensor("beta (CRITICAL - should be bounded!)", outputs['beta'], mask_expanded)
    analyze_tensor("conditioned_film (gamma * x + beta)", outputs['conditioned_film'], mask_expanded)
    analyze_tensor("conditioned (final blend)", outputs['conditioned'], mask_expanded)
    
    # Check masking
    print("\n" + "="*80)
    print("9. MASKING ANALYSIS")
    print("="*80)
    
    print("\n  is_entity values by position:")
    is_entity_flat = outputs['is_entity'][0, :, 0]
    for i, val in enumerate(is_entity_flat.tolist()):
        pos_type = "TEXT" if text_only_mask[0, i] else "SPECIAL"
        marker = "*" if val > 0.5 else " "
        print(f"  {marker} Position {i:2d} ({pos_type:7s}): is_entity = {val:.4f}")
    
    print("\n  ISSUE: For SPECIAL positions, is_entity should be ~0 (not modified)")
    print("  But we see high is_entity values for header positions!")
    
    # Check if masking would fix
    print("\n10. SIMULATION: What if we apply text_only_mask to conditioned?")
    text_only_mask_expanded = text_only_mask.unsqueeze(-1).float()
    conditioned_fixed = outputs['original_embeddings'] * (1 - text_only_mask_expanded) + outputs['conditioned'] * text_only_mask_expanded
    
    analyze_tensor("conditioned_fixed (with masking)", conditioned_fixed, mask_expanded)
    
    print("\n11. COMPARISON: Original vs Conditioned vs Fixed")
    print("  Position | Original Range | Conditioned Range | Fixed Range | is_entity")
    print("  " + "-"*80)
    for i in range(min(10, input_ids.shape[1])):
        orig = outputs['original_embeddings'][0, i]
        cond = outputs['conditioned'][0, i]
        fixed = conditioned_fixed[0, i]
        ie = outputs['is_entity'][0, i, 0].item()
        pos_type = "TEXT" if text_only_mask[0, i] else "SPEC"
        print(f"  {i:2d} ({pos_type}) | [{orig.min():.3f}, {orig.max():.3f}] | [{cond.min():.3f}, {cond.max():.3f}] | [{fixed.min():.3f}, {fixed.max():.3f}] | {ie:.3f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"""
Issues Found:
1. Beta is UNBOUNDED: range [{outputs['beta'].min().item():.2f}, {outputs['beta'].max().item():.2f}]
   - Should be bounded to ~[-0.2, 0.2] to match embedding scale
   
2. is_entity for SPECIAL positions is HIGH (should be ~0):
   - Header positions (0-2): is_entity ≈ 1.0
   - Pad positions (3-6): is_entity ≈ 1.0
   - This means special token embeddings are completely replaced by FiLM output!
   
3. Conditioned embeddings have 50x larger range than original:
   - Original: ~[-0.2, 0.2]
   - Conditioned: ~[-7, 8]
   
Fixes Needed (already applied to training script):
1. Bound beta: beta = 0.2 * torch.tanh(beta)
2. Apply text_only_mask: conditioned = original * (1 - mask) + conditioned * mask
""")


if __name__ == "__main__":
    main()
