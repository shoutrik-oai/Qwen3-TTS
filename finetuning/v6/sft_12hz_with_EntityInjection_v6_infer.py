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
from sft_12hz_with_EntityInjection_v6 import EntityInjectionModuleConfig, EntityInjectionModule


# class InjectedTextEmbedding(nn.Module):
    # """
    # Wrapper around text_embedding that applies entity injection.
    # """
    # def __init__(self, orig_embed, entity_mod, target_dtype):
    #     super().__init__()
    #     self.orig_embed = orig_embed
    #     self.entity_mod = entity_mod
    #     self.target_dtype = target_dtype
    
    # def forward(self, input_ids):
    #     # Get original embeddings
    #     embeddings = self.orig_embed(input_ids)
    #     # Apply entity injection (returns conditioned, entity_logits, entity_detection_logits)
    #     conditioned, _, _ = self.entity_mod(embeddings)
    #     return conditioned.to(self.target_dtype)


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
        config
    ):
        self.qwen3tts = qwen3tts  # The full wrapper with .processor
        self.entity_injection_module = entity_injection_module
        self.entity_type_to_index = entity_type_to_index
        self.index_to_entity_type = index_to_entity_type
        
        # Store original get_text_embeddings for hooking
        self._original_get_text_embeddings = self.qwen3tts.model.talker.get_text_embeddings
        self._injected_embedding = None
        self._entity_injection_enabled = False
        self.config = config
    
    # def _create_injected_embedding(self):
    #     """Create the injected text embedding module."""
    #     original_embedding = self._original_get_text_embeddings()
    #     dtype = next(self.qwen3tts.model.parameters()).dtype
    #     device = next(self.qwen3tts.model.parameters()).device
        
    #     injected = InjectedTextEmbedding(
    #         original_embedding,
    #         self.entity_injection_module,
    #         dtype
    #     ).to(device)
        
    #     return injected
    
    # def _get_text_embeddings_with_injection(self):
    #     """Returns the injected text embedding module."""
    #     if self._injected_embedding is None:
    #         self._injected_embedding = self._create_injected_embedding()
    #     return self._injected_embedding
    
    # def enable_entity_injection(self):
    #     """Enable entity injection during generation."""
    #     self._entity_injection_enabled = True
    #     self.qwen3tts.model.talker.get_text_embeddings = self._get_text_embeddings_with_injection
    
    # def disable_entity_injection(self):
    #     """Disable entity injection (use original model)."""
    #     self._entity_injection_enabled = False
    #     self.qwen3tts.model.talker.get_text_embeddings = self._original_get_text_embeddings
    
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
            conditioned, entity_logits, entity_detection_logits, gamma, beta, _ = self.extract_conditioned_embeddings(text)
        else:
            conditioned = None  # Use original embeddings

        wavs, sr = self.qwen3tts.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language,
            conditioned=conditioned,
            non_streaming_mode=True,
            max_new_tokens=kwargs.pop('max_new_tokens', 512),  # Limit generation length
            **kwargs,
        )
        return wavs, sr
            
        # finally:
        #     self.disable_entity_injection()

    @torch.no_grad()
    def extract_conditioned_embeddings(self, text: str) -> torch.Tensor:
        device = next(self.qwen3tts.model.parameters()).device
        
        # Tokenize the same way as training: include special tokens then strip
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n" # 3 + 10 + 5 = 18
        full_input_ids = self.qwen3tts.processor.tokenizer(
            assistant_text,
            return_tensors="pt"
        ).input_ids.to(device)  # [1, T]

        print(f"full_input_ids: {full_input_ids}")
        
        # Match training preprocessing: strip last 5 tokens (<|tts_eos|><|im_end|>)
        full_input_ids = full_input_ids[:, :-5] # 18 - 5 = 13
        text_ids_len = full_input_ids.shape[1] # 13
        
        # Get config from qwen3tts mod
        
        # Build input_ids with padding structure matching training
        # Structure: [first 3 tokens] [4 pad tokens] [tts_bos] [remaining text tokens]
        input_ids = torch.zeros((1, text_ids_len + 6), device=device, dtype=torch.long) # 13 + 8 = 21
        input_ids[0, :3] = full_input_ids[0, :3] # 0-2
        input_ids[0, 3:7] = self.config.tts_pad_token_id # 3-6
        input_ids[0, 7] = self.config.tts_bos_token_id # 7
        input_ids[0, 8:8 + text_ids_len - 3] = full_input_ids[0, 3:] # 8-17
        input_ids[0, 8+text_ids_len-3] = self.config.tts_eos_token_id # 18

        print(f"input_ids: {input_ids}")
        
        # Get text embeddings
        text_embed = self.qwen3tts.model.talker.model.text_embedding(input_ids)
        B, T, D = text_embed.shape
        text_only_mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        text_only_mask[0, 8:8+text_ids_len-3] = True
        
        # Get entity predictions (returns conditioned, entity_logits, entity_detection_logits, gamma, beta)
        conditioned, entity_logits, entity_detection_logits, gamma, beta = self.entity_injection_module(
            text_embed, text_mask=text_only_mask, return_details=True
        )
        
        # Only apply entity injection to text positions, preserve original embeddings for header/special tokens
        text_only_mask_expanded = text_only_mask.unsqueeze(-1).float()
        conditioned = text_embed * (1 - text_only_mask_expanded) + conditioned * text_only_mask_expanded
        
        # Debug: Check for numerical issues
        print(f"=== DEBUG: Embedding Analysis ===")
        print(f"text_embed dtype: {text_embed.dtype}, range: [{text_embed.min().item():.4f}, {text_embed.max().item():.4f}]")
        print(f"conditioned dtype: {conditioned.dtype}, range: [{conditioned.min().item():.4f}, {conditioned.max().item():.4f}]")
        print(f"gamma range: [{gamma.min().item():.4f}, {gamma.max().item():.4f}]")
        print(f"beta range: [{beta.min().item():.4f}, {beta.max().item():.4f}]")
        print(f"has NaN: text_embed={text_embed.isnan().any()}, conditioned={conditioned.isnan().any()}")
        print(f"has Inf: text_embed={text_embed.isinf().any()}, conditioned={conditioned.isinf().any()}")
        
        # Convert to model dtype (BFloat16)
        model_dtype = next(self.qwen3tts.model.parameters()).dtype
        conditioned = conditioned.to(model_dtype)
        
        print(f"conditioned: {conditioned.shape}")  
        print(f"entity_detection_logits: {entity_detection_logits}")
        return conditioned, entity_logits, entity_detection_logits, gamma, beta, full_input_ids

    
    @torch.no_grad()
    def predict_entities(self, entity_logits, entity_detection_logits, gamma, beta, full_input_ids) -> List[dict]:
        """
        Predict entity types for a given text (for debugging/analysis).
        
        Returns predictions including:
        - Entity type classification
        - Entity detection (is_entity probability)
        """
        self.device = next(self.qwen3tts.model.parameters()).device
        
        # Decode predictions
        predictions = []
        # Decode tokens from the restructured input_ids (matches prediction indices)
        all_tokens = self.qwen3tts.processor.tokenizer.convert_ids_to_tokens(full_input_ids[0].tolist())
        print(all_tokens)

        for idx, token in enumerate(all_tokens[3:]):
            logit_ = entity_logits[0, 8+idx]
            type_prob = torch.softmax(logit_, dim=-1)
            pred_idx = type_prob.argmax().item()
            pred_type = self.index_to_entity_type.get(pred_idx, "UNKNOWN")
            is_entity_prob = entity_detection_logits[0, 8+idx]
            confidence = type_prob[pred_idx].item()

            pred_type = pred_type if is_entity_prob > 0.5 else "PLAIN_WORD"

            predictions.append({
                "position": idx,  # 0-indexed from actual text start
                "token": token,
                "predicted_type": pred_type,
                "type_confidence": type_prob[pred_idx].item(),
                "is_entity_prob": is_entity_prob.item(),
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
    # Note: entity_prob defaults to 0.0 which is fine for inference (only affects init)
    entity_injection_config = EntityInjectionModuleConfig(
        hidden_size=entity_config_dict["hidden_size"],
        num_entities=entity_config_dict["num_entities"],
        num_layers=entity_config_dict.get("num_layers", 2),
        num_heads=entity_config_dict.get("num_heads", 8),
        dim_feedforward=entity_config_dict.get("dim_feedforward", 2048),
        dropout=entity_config_dict.get("dropout", 0.1),
        entity_prob=entity_config_dict.get("entity_prob", 0.0),  # Only affects init
    )
    entity_injection_module = EntityInjectionModule(entity_injection_config)
    
    # Load entity injection module weights
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
    config = AutoConfig.from_pretrained(model_path)
    
    # Create the combined wrapper
    model = Qwen3TTSWithEntityInjection(
        qwen3tts=qwen3tts,
        entity_injection_module=entity_injection_module,
        entity_type_to_index=entity_type_to_index,
        index_to_entity_type=index_to_entity_type,
        config=config,
    )
    
    print(f"Model loaded successfully!")
    print(f"  - Entity types: {len(entity_type_to_index)}")
    print(f"  - Entity encoder layers: {entity_injection_config.num_layers}")
    print(f"  - Entity detector bias: {entity_injection_module.entity_detector.bias.item():.4f}")
    
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
    print("\n=== Entity Predictions ===")
    _, entity_logits, entity_detection_logits, gamma, beta, full_input_ids = model.extract_conditioned_embeddings(args.text)
    predictions = model.predict_entities(entity_logits, entity_detection_logits, gamma, beta, full_input_ids)
    print(f"  {'Pos':>3s}  {'Token':20s}  {'Type':20s}  {'TypeConf':>8s}  {'IsEntityProb':>8s}  {'IsEntity':>8s}")
    print("  " + "-" * 70)
    for pred in predictions:
        # Show all tokens, highlight entities (is_entity_prob > 0.5)
        is_entity = torch.sigmoid(torch.tensor(pred["is_entity_prob"])).item()
        marker = "*" if is_entity>0.5 else " "
        print(f"{marker} {pred['position']:3d}  {pred['token']:20s}  {pred['predicted_type']:20s}  {pred['type_confidence']:.4f}    {pred['is_entity_prob']:.4f}    {is_entity:.4f}")
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
