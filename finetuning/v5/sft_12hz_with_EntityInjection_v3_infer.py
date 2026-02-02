# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Inference script for Qwen3-TTS with Entity Injection Module

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel
from safetensors.torch import load_file
from transformers import AutoConfig


@dataclass
class EntityInjectionModuleConfig:
    hidden_size: int
    num_entities: int
    num_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    batch_first: bool = True


class EntityInjectionModule(nn.Module):
    """
    Learns entity-aware representations from text embeddings.
    """
    
    def __init__(self, config: EntityInjectionModuleConfig):
        super().__init__()
        
        hidden_size = config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=config.batch_first
        )
        self.entity_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.num_entities = config.num_entities
        self.type_head = nn.Linear(hidden_size, self.num_entities)
        self.entity_delta_proj = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Parameter(torch.zeros(1) + 0.1)
    
    def forward(self, text_embeddings, text_mask=None, return_type_logits=False):
        if text_mask is not None:
            src_key_padding_mask = (text_mask == 0)
        else:
            src_key_padding_mask = None
        
        type_hidden = self.entity_encoder(text_embeddings, src_key_padding_mask=src_key_padding_mask)
        type_delta = self.entity_delta_proj(type_hidden)
        type_enriched = text_embeddings + self.gate * type_delta
        
        if return_type_logits:
            type_logits = self.type_head(type_enriched)
            return type_enriched, type_logits
        
        return type_enriched


class InjectedTextEmbedding(nn.Module):
    """
    Wrapper around text_embedding that applies entity injection.
    """
    def __init__(self, orig_embed, entity_mod, target_dtype):
        super().__init__()
        self.orig_embed = orig_embed
        self.entity_mod = entity_mod
        self.target_dtype = target_dtype
    
    def forward(self, input_ids):
        # Get original embeddings
        embeddings = self.orig_embed(input_ids)
        # Apply entity injection
        enriched = self.entity_mod(embeddings)
        return enriched.to(self.target_dtype)


class Qwen3TTSWithEntityInjection:
    """
    Wrapper that adds entity injection to Qwen3TTSModel.
    
    This keeps the original Qwen3TTSModel interface intact (with .processor, 
    .generate_custom_voice, etc.) while injecting entity information.
    """
    
    def __init__(
        self,
        qwen3tts: Qwen3TTSModel,
        entity_injection_module: EntityInjectionModule,
        entity_type_to_index: dict,
        index_to_entity_type: dict,
    ):
        self.qwen3tts = qwen3tts  # The full wrapper with .processor
        self.entity_injection_module = entity_injection_module
        self.entity_type_to_index = entity_type_to_index
        self.index_to_entity_type = index_to_entity_type
        
        # Store original get_text_embeddings for hooking
        self._original_get_text_embeddings = self.qwen3tts.model.talker.get_text_embeddings
        self._injected_embedding = None
        self._entity_injection_enabled = False
    
    def _create_injected_embedding(self):
        """Create the injected text embedding module."""
        original_embedding = self._original_get_text_embeddings()
        dtype = next(self.qwen3tts.model.parameters()).dtype
        device = next(self.qwen3tts.model.parameters()).device
        
        injected = InjectedTextEmbedding(
            original_embedding,
            self.entity_injection_module,
            dtype
        ).to(device)
        
        return injected
    
    def _get_text_embeddings_with_injection(self):
        """Returns the injected text embedding module."""
        if self._injected_embedding is None:
            self._injected_embedding = self._create_injected_embedding()
        return self._injected_embedding
    
    def enable_entity_injection(self):
        """Enable entity injection during generation."""
        self._entity_injection_enabled = True
        self.qwen3tts.model.talker.get_text_embeddings = self._get_text_embeddings_with_injection
    
    def disable_entity_injection(self):
        """Disable entity injection (use original model)."""
        self._entity_injection_enabled = False
        self.qwen3tts.model.talker.get_text_embeddings = self._original_get_text_embeddings
    
    @torch.no_grad()
    def generate_custom_voice(
        self,
        text: str,
        speaker: str,
        language: str = "Auto",
        use_entity_injection: bool = True,
        **kwargs,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Generate speech with entity injection using custom voice.
        """
        if use_entity_injection:
            self.enable_entity_injection()
        
        try:
            # Use the Qwen3TTSModel's generate_custom_voice method directly
            wavs, sr = self.qwen3tts.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language,
                **kwargs,
            )
            return wavs, sr
            
        finally:
            self.disable_entity_injection()
    
    @torch.no_grad()
    def predict_entities(self, text: str) -> List[dict]:
        """
        Predict entity types for a given text (for debugging/analysis).
        """
        device = next(self.qwen3tts.model.parameters()).device
        
        assistant_text = f"<|im_start|>assistant\n<|tts_bos|>{text}<|tts_eos|><|im_end|>"
        input_ids = self.qwen3tts.processor.tokenizer(
            assistant_text,
            return_tensors="pt"
        ).input_ids.to(device)
        
        # Get text embeddings
        text_embed = self.qwen3tts.model.talker.model.text_embedding(input_ids)
        
        # Get entity predictions
        _, entity_logits = self.entity_injection_module(text_embed, return_type_logits=True)
        
        # Decode predictions
        predictions = []
        tokens = self.qwen3tts.processor.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        probs = torch.softmax(entity_logits[0], dim=-1)
        
        for i, (token, prob) in enumerate(zip(tokens, probs)):
            pred_idx = prob.argmax().item()
            pred_type = self.index_to_entity_type.get(pred_idx, "UNKNOWN")
            predictions.append({
                "position": i,
                "token": token,
                "predicted_type": pred_type,
                "confidence": prob[pred_idx].item(),
            })
        
        return predictions


def load_model(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16
) -> Qwen3TTSWithEntityInjection:
    """
    Load the finetuned model with entity injection module.
    
    Args:
        model_path: Path to the checkpoint directory
        device: Device to load model on
        dtype: Model dtype
        
    Returns:
        Qwen3TTSWithEntityInjection wrapper
    """
    print(f"Loading Qwen3TTS from {model_path}...")
    
    # Load the full Qwen3TTSModel wrapper (has .model, .processor, generation methods)
    qwen3tts = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map=device,
        dtype=dtype,
        attn_implementation="sdpa",
    )
    
    # Load entity type mapping
    entity_mapping_path = os.path.join(model_path, "entity_type_mapping.json")
    if os.path.exists(entity_mapping_path):
        with open(entity_mapping_path, 'r') as f:
            entity_mapping = json.load(f)
        entity_type_to_index = entity_mapping["entity_type_to_index"]
        index_to_entity_type = {int(k): v for k, v in entity_mapping["index_to_entity_type"].items()}
        entity_config_dict = entity_mapping["entity_injection_config"]
    else:
        raise FileNotFoundError(f"Entity type mapping not found at {entity_mapping_path}")
    
    # Create and load EntityInjectionModule
    entity_injection_config = EntityInjectionModuleConfig(
        hidden_size=entity_config_dict["hidden_size"],
        num_entities=entity_config_dict["num_entities"],
        num_layers=entity_config_dict["num_layers"],
        num_heads=entity_config_dict["num_heads"],
        dim_feedforward=entity_config_dict["dim_feedforward"],
        dropout=entity_config_dict["dropout"],
        activation=entity_config_dict["activation"],
        batch_first=entity_config_dict["batch_first"],
    )
    entity_injection_module = EntityInjectionModule(entity_injection_config)
    
    entity_module_path = os.path.join(model_path, "entity_injection_module.safetensors")
    if os.path.exists(entity_module_path):
        print(f"Loading EntityInjectionModule from {entity_module_path}...")
        entity_state_dict = load_file(entity_module_path)
        entity_injection_module.load_state_dict(entity_state_dict)
    else:
        raise FileNotFoundError(f"Entity injection module not found at {entity_module_path}")
    
    # Move to correct device/dtype
    model_device = next(qwen3tts.model.parameters()).device
    model_dtype = next(qwen3tts.model.parameters()).dtype
    entity_injection_module = entity_injection_module.to(model_device).to(model_dtype)
    entity_injection_module.eval()
    
    # Create the combined wrapper
    model = Qwen3TTSWithEntityInjection(
        qwen3tts=qwen3tts,
        entity_injection_module=entity_injection_module,
        entity_type_to_index=entity_type_to_index,
        index_to_entity_type=index_to_entity_type,
    )
    
    print(f"Model loaded successfully!")
    print(f"  - Entity types: {len(entity_type_to_index)}")
    print(f"  - Gate value: {entity_injection_module.gate.item():.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Inference with Entity Injection")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the finetuned checkpoint directory")
    parser.add_argument("--text", type=str, required=True,
                       help="Text to synthesize")
    parser.add_argument("--speaker", type=str, required=True,
                       help="Speaker name (as configured during training)")
    parser.add_argument("--language", type=str, default="Auto",
                       help="Language (English, Chinese, Auto)")
    parser.add_argument("--output", type=str, default="output.wav",
                       help="Output audio file path")
    parser.add_argument("--no_entity_injection", action="store_true",
                       help="Disable entity injection (use base model only)")
    parser.add_argument("--show_entities", action="store_true",
                       help="Show predicted entity types for the input text")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.9,
                       help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=1.0,
                       help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.05,
                       help="Repetition penalty")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    model = load_model(model_path=args.model_path, device=device)
    
    # Show entity predictions if requested
    if args.show_entities:
        print("\n=== Entity Predictions ===")
        predictions = model.predict_entities(args.text)
        for pred in predictions:
            if pred["predicted_type"] not in ["PLAIN_WORD", "PLAIN", "SPECIAL"]:
                print(f"  {pred['position']:3d}: {pred['token']:20s} -> {pred['predicted_type']:15s} ({pred['confidence']:.3f})")
        print()
    
    # Generate audio
    print(f"\nGenerating audio for: \"{args.text}\"")
    print(f"  Speaker: {args.speaker}")
    print(f"  Language: {args.language}")
    print(f"  Entity Injection: {'Disabled' if args.no_entity_injection else 'Enabled'}")
    
    wavs, sample_rate = model.generate_custom_voice(
        text=args.text,
        speaker=args.speaker,
        language=args.language,
        use_entity_injection=not args.no_entity_injection,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )
    
    audio = wavs[0]
    
    # Save audio
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    sf.write(args.output, audio, sample_rate)
    
    print(f"\nGenerated audio: {len(audio)/sample_rate:.2f} seconds @ {sample_rate} Hz")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
