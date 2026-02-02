# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Limit CPU threads BEFORE importing torch/numpy
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_MAX_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import os
import shutil
import warnings

# Silence Flash Attention warnings
warnings.filterwarnings("ignore", message=".*Flash Attention.*")

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)
import torch.nn as nn
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoModel, AutoConfig as HFAutoConfig


@dataclass
class EntityInjectionModuleConfig:
    hidden_size: int  # Qwen's text hidden size
    num_entities: int
    # Pretrained encoder settings
    pretrained_encoder: str = "distilbert-base-uncased"  # or "bert-base-uncased", "prajjwal1/bert-tiny", etc.
    freeze_encoder: bool = False  # If True, only train projections + FiLM
    # Fallback to scratch training if pretrained_encoder is None
    num_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1


class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Applies: H_out = gamma * H_in + beta
    
    Unlike gating (H + g*delta), FiLM can't collapse to identity.
    gamma ≈ 1, beta ≈ 0 initially, but always structurally modifies H.
    """
    
    def __init__(self, hidden_size, num_entities):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_entities = num_entities
        
        # Learnable entity type embeddings
        self.entity_type_embeddings = nn.Embedding(num_entities, hidden_size)
        
        # MLP to generate gamma and beta from entity embedding
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),  # outputs [gamma_offset, beta]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize entity embeddings
        nn.init.normal_(self.entity_type_embeddings.weight, std=0.02)
        
        # Initialize MLP - crucial for near-identity initialization
        # The final layer should output near-zero so gamma ≈ 1, beta ≈ 0
        for module in self.film_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        
        # Make final layer output very small values initially
        final_layer = self.film_mlp[-1]
        nn.init.normal_(final_layer.weight, std=0.001)
        nn.init.zeros_(final_layer.bias)
    
    def forward(self, hidden_states, entity_logits, temperature=1.0):
        """
        Args:
            hidden_states: [B, T, D] - text hidden states to condition
            entity_logits: [B, T, num_entities] - predicted entity type logits
            temperature: softmax temperature (lower = sharper)
            
        Returns:
            conditioned: [B, T, D] - FiLM-conditioned hidden states
            gamma: [B, T, D] - scale factors (for logging)
            beta: [B, T, D] - shift factors (for logging)
        """
        # Soft entity type embedding via attention over entity types
        type_probs = F.softmax(entity_logits / temperature, dim=-1)  # [B, T, num_entities]
        type_emb = type_probs @ self.entity_type_embeddings.weight   # [B, T, D]
        
        # Generate FiLM parameters
        film_params = self.film_mlp(type_emb)  # [B, T, 2*D]
        gamma_offset, beta = film_params.chunk(2, dim=-1)  # each [B, T, D]
        
        # gamma = 1 + offset (so initial gamma ≈ 1)
        # Using tanh to bound gamma_offset to [-1, 1] for stability
        gamma = 1.0 + 0.5 * torch.tanh(gamma_offset)  # gamma in [0.5, 1.5]
        
        # Apply FiLM: scale and shift
        conditioned = gamma * hidden_states + beta
        
        return conditioned, gamma, beta


class EntityInjectionModule(nn.Module):
    """
    Entity-aware conditioning using FiLM (Feature-wise Linear Modulation).
    
    Architecture:
        1. Entity Encoder: Pretrained BERT/DistilBERT (finetuned with low LR)
           - Input projection: Qwen hidden → BERT hidden
           - Output projection: BERT hidden → Qwen hidden
        2. Entity Head: Predicts entity types per token (trained from scratch)
        3. FiLM Layer: Conditions embeddings using soft entity predictions (trained from scratch)
    
    Key difference from gate-based approach:
        - Gate: H + g * delta → can collapse (g → 0)
        - FiLM: γ * H + β → structurally always modifies H (γ ≈ 1)
    
    Input:
        text_embeddings: [B, T, D] - text token embeddings
        text_mask: [B, T] - 1 for valid text positions, 0 for padding
        
    Output:
        conditioned: [B, T, D] - entity-conditioned embeddings
        entity_logits: [B, T, num_entities] - entity type predictions
    """
    
    def __init__(self, config: EntityInjectionModuleConfig):
        super().__init__()
        
        qwen_hidden_size = config.hidden_size
        num_entities = config.num_entities
        self.num_entities = num_entities
        self.use_pretrained = config.pretrained_encoder is not None
        
        if self.use_pretrained:
            # Load pretrained encoder (e.g., DistilBERT, BERT-tiny)
            print(f"Loading pretrained encoder: {config.pretrained_encoder}")
            self.pretrained_encoder = AutoModel.from_pretrained(config.pretrained_encoder)
            bert_hidden_size = self.pretrained_encoder.config.hidden_size
            
            # Projection layers: Qwen hidden <-> BERT hidden
            self.input_proj = nn.Linear(qwen_hidden_size, bert_hidden_size)
            self.output_proj = nn.Linear(bert_hidden_size, qwen_hidden_size)
            
            # Optionally freeze the pretrained encoder
            if config.freeze_encoder:
                print("Freezing pretrained encoder weights")
                for param in self.pretrained_encoder.parameters():
                    param.requires_grad = False
            
            # Output dimension for entity head
            encoder_output_dim = qwen_hidden_size
        else:
            # Fallback: train entity encoder from scratch
            print("Training entity encoder from scratch")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=qwen_hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True
            )
            self.entity_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
            self.input_proj = None
            self.output_proj = None
            encoder_output_dim = qwen_hidden_size
        
        # Entity type classification head (trained from scratch)
        self.entity_head = nn.Linear(encoder_output_dim, num_entities)
        
        # FiLM conditioning layer (trained from scratch)
        self.film_layer = FiLMLayer(qwen_hidden_size, num_entities)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_head.weight)
        nn.init.zeros_(self.entity_head.bias)
        
        if self.input_proj is not None:
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.zeros_(self.input_proj.bias)
        if self.output_proj is not None:
            nn.init.xavier_uniform_(self.output_proj.weight)
            nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, text_embeddings, text_mask=None, return_details=False):
        """
        Two-stage forward pass:
            Stage 1: Entity prediction from text embeddings (via pretrained encoder)
            Stage 2: FiLM conditioning using entity predictions
        
        Args:
            text_embeddings: [B, T, D] text token embeddings from Qwen
            text_mask: [B, T] attention mask (1 = valid, 0 = padding)
            return_details: whether to return additional info for logging/loss
            
        Returns:
            conditioned: [B, T, D] entity-conditioned embeddings
            entity_logits: [B, T, num_entities] entity type predictions
            (gamma, beta): FiLM parameters if return_details=True
        """
        if self.use_pretrained:
            # ===== PRETRAINED ENCODER PATH =====
            # Project Qwen embeddings to BERT dimension
            projected = self.input_proj(text_embeddings)  # [B, T, D_bert]
            
            # Create attention mask for BERT (1 = attend, 0 = ignore)
            if text_mask is not None:
                attention_mask = text_mask.long()
            else:
                attention_mask = torch.ones(text_embeddings.shape[:2], device=text_embeddings.device)
            
            # Pass through pretrained encoder
            # BERT/DistilBERT expects attention_mask where 1 = attend
            encoder_outputs = self.pretrained_encoder(
                inputs_embeds=projected,
                attention_mask=attention_mask,
                return_dict=True
            )
            entity_hidden_bert = encoder_outputs.last_hidden_state  # [B, T, D_bert]
            
            # Project back to Qwen dimension
            entity_hidden = self.output_proj(entity_hidden_bert)  # [B, T, D_qwen]
        else:
            # ===== SCRATCH ENCODER PATH =====
            if text_mask is not None:
                src_key_padding_mask = (text_mask == 0)
            else:
                src_key_padding_mask = None
            
            entity_hidden = self.entity_encoder(text_embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # ===== STAGE 1: Entity Prediction =====
        entity_logits = self.entity_head(entity_hidden)  # [B, T, num_entities]
        
        # ===== STAGE 2: FiLM Conditioning =====
        # Apply FiLM using soft entity predictions
        conditioned, gamma, beta = self.film_layer(entity_hidden, entity_logits)
        
        if return_details:
            return conditioned, entity_logits, gamma, beta
        
        return conditioned, entity_logits


class Qwen3TTSModelWithEntityInjection(nn.Module):
    def __init__(self, base_model, qwen3_config, entity_injection_module, special_indices):
        super().__init__()
        self.base_model = base_model
        self.qwen3_config = qwen3_config
        self.entity_injection_module = entity_injection_module
        self.special_indices = special_indices
        
    def forward(self, batch):
        input_ids = batch['input_ids']
        codec_ids = batch['codec_ids']
        ref_mels = batch['ref_mels']
        text_embedding_mask = batch['text_embedding_mask']
        codec_embedding_mask = batch['codec_embedding_mask']
        attention_mask = batch['attention_mask']
        codec_0_labels = batch['codec_0_labels']
        codec_mask = batch['codec_mask']
        entities = batch['entities']  # [B, T, 1] with entity labels (-100 for non-entities)

        target_speaker_embedding = self.base_model.speaker_encoder(ref_mels).detach()

        input_text_ids = input_ids[:, :, 0]
        input_codec_ids = input_ids[:, :, 1]

        # Get base text embeddings
        input_text_embedding = self.base_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        text_only_mask = batch['text_only_mask']
        
        # ===== Entity Injection with FiLM =====
        # Two-stage: 1) Predict entities, 2) Condition with FiLM
        conditioned_embedding, entity_logits, gamma, beta = self.entity_injection_module(
            input_text_embedding, 
            text_mask=text_only_mask.long(),
            return_details=True
        )
        
        # Continue through text projection
        conditioned_embedding = conditioned_embedding.to(torch.bfloat16)
        input_text_embedding = self.base_model.talker.text_projection(conditioned_embedding)

        # ===== Entity Loss =====
        entity_labels = entities[:, :, 0].clone()  # [B, T]
        
        # Mask out special indices (SPECIAL, PLAIN_WORD, PLAIN) - ignore them
        special_mask = torch.isin(entity_labels, self.special_indices.to(entity_labels.device))
        entity_labels[special_mask] = -100
        
        entity_type_loss = F.cross_entropy(
            entity_logits.view(-1, entity_logits.size(-1)),
            entity_labels.view(-1),
            ignore_index=-100,
        )
        
        # DDP FIX: Ensure ALL parameters participate in loss computation
        # Include gamma, beta sums with 0 coefficient to keep FiLM layer in graph
        # This ensures entity_type_embeddings and film_mlp always receive gradients
        entity_type_loss = entity_type_loss + 0.0 * gamma.sum() + 0.0 * beta.sum()
        
        # DDP FIX: When using inputs_embeds, BERT's word_embeddings are bypassed
        # Add dummy term to ensure word_embeddings weights participate in gradient computation
        if hasattr(self.entity_injection_module, 'pretrained_encoder'):
            # For DistilBERT/BERT, the word embeddings are at embeddings.word_embeddings
            word_emb_weight = self.entity_injection_module.pretrained_encoder.embeddings.word_embeddings.weight
            entity_type_loss = entity_type_loss + 0.0 * word_emb_weight.sum()
    
        if torch.isnan(entity_type_loss):
            entity_type_loss = 0.0 * entity_logits.sum() + 0.0 * gamma.sum() + 0.0 * beta.sum()
            # Also include word_embeddings in NaN fallback for DDP
            if hasattr(self.entity_injection_module, 'pretrained_encoder'):
                word_emb_weight = self.entity_injection_module.pretrained_encoder.embeddings.word_embeddings.weight
                entity_type_loss = entity_type_loss + 0.0 * word_emb_weight.sum()
        
        # ===== Continue with TTS Forward =====
        input_codec_embedding = self.base_model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
        input_codec_embedding[:, 6, :] = target_speaker_embedding

        input_embeddings = input_text_embedding + input_codec_embedding

        for i in range(1, 16):
            codec_i_embedding = self.base_model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
            codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
            input_embeddings = input_embeddings + codec_i_embedding

        outputs = self.base_model.talker(
            inputs_embeds=input_embeddings[:, :-1, :],
            attention_mask=attention_mask[:, :-1],
            labels=codec_0_labels[:, 1:],
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[0][-1]
        talker_hidden_states = hidden_states[codec_mask[:, 1:]]
        talker_codec_ids = codec_ids[codec_mask]

        sub_talker_logits, sub_talker_loss = self.base_model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)
        
        # FiLM statistics for logging
        gamma_mean = gamma.mean()
        beta_mean = beta.mean()

        return {
            'loss': outputs.loss,
            'sub_talker_loss': sub_talker_loss,
            'entity_type_loss': entity_type_loss,
            'gamma_mean': gamma_mean,
            'beta_mean': beta_mean,
            'target_speaker_embedding': target_speaker_embedding,
        }

def get_optimizer_and_scheduler(model, args, num_epochs, num_batches, accum_grad):
    """
    Optimizer for FiLM-based entity injection with pretrained encoder.
    
    Parameter groups:
        1. Base TTS model - base LR
        2. Pretrained encoder (BERT/DistilBERT) - very low LR (0.1x base)
        3. Projection layers - higher LR (10x base)
        4. Entity head - higher LR (10x base)
        5. FiLM layer - higher LR (10x base)
    """
    base_model = model.base_model
    entity_module = model.entity_injection_module

    # Categorize parameters
    pretrained_encoder_params = []
    projection_params = []
    entity_head_params = []
    film_layer_params = []
    scratch_encoder_params = []
    
    for name, param in entity_module.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen params
        
        if "pretrained_encoder" in name:
            pretrained_encoder_params.append(param)
        elif "input_proj" in name or "output_proj" in name:
            projection_params.append(param)
        elif "entity_head" in name:
            entity_head_params.append(param)
        elif "film_layer" in name:
            film_layer_params.append(param)
        elif "entity_encoder" in name:
            scratch_encoder_params.append(param)

    # Learning rates
    base_lr = args.lr
    pretrained_lr = args.lr * 0.1   # Very low LR for pretrained (finetune gently)
    new_module_lr = args.lr * 10    # Higher LR for new modules (train from scratch)

    optimizer_groups = [
        {
            "params": list(base_model.parameters()),
            "lr": base_lr,
            "weight_decay": 0.01,
            "name": "base_tts_model",
        },
    ]
    
    # Add pretrained encoder params (if any) with low LR
    if pretrained_encoder_params:
        optimizer_groups.append({
            "params": pretrained_encoder_params,
            "lr": pretrained_lr,
            "weight_decay": 0.01,
            "name": "pretrained_encoder",
        })
    
    # Add scratch encoder params (if any) with high LR
    if scratch_encoder_params:
        optimizer_groups.append({
            "params": scratch_encoder_params,
            "lr": new_module_lr,
            "weight_decay": 0.01,
            "name": "scratch_encoder",
        })
    
    # Add projection params with high LR
    if projection_params:
        optimizer_groups.append({
            "params": projection_params,
            "lr": new_module_lr,
            "weight_decay": 0.01,
            "name": "projections",
        })
    
    # Add entity head params with high LR
    if entity_head_params:
        optimizer_groups.append({
            "params": entity_head_params,
            "lr": new_module_lr,
            "weight_decay": 0.01,
            "name": "entity_head",
        })
    
    # Add FiLM layer params with high LR
    if film_layer_params:
        optimizer_groups.append({
            "params": film_layer_params,
            "lr": new_module_lr,
            "weight_decay": 0.01,
            "name": "film_layer",
        })

    optimizer = AdamW(optimizer_groups)

    # All groups use same LR schedule (just different base LRs)
    num_groups = len(optimizer_groups)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[lambda step: 1.0] * num_groups  # Constant LR (base LRs already differentiated)
    )
    
    return optimizer, scheduler

def check_parameter_changes(accelerator, model, params_before, print_details=False):
    entity_module = accelerator.unwrap_model(model).entity_injection_module
    params_not_updating = []
    
    for name, p in entity_module.named_parameters():
        diff = torch.abs(p - params_before[name]).max().item()
        if diff < 1e-8:
            params_not_updating.append(name)
    
    if print_details and accelerator.is_main_process:
        if params_not_updating:
            accelerator.print("--- Params NOT updating ---")
            for name in params_not_updating:
                accelerator.print(f"  {name}")
    
    all_updating = len(params_not_updating) == 0
    return all_updating, params_not_updating

def train():

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--entity_loss_weight", type=float, default=0.1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    # Pretrained encoder settings
    parser.add_argument("--pretrained_encoder", type=str, default="distilbert-base-uncased",
                        help="Pretrained encoder for entity detection. Options: distilbert-base-uncased, "
                             "prajjwal1/bert-tiny, prajjwal1/bert-mini, bert-base-uncased, or 'none' for scratch")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze pretrained encoder weights (only train projections + FiLM)")
    # Wandb arguments
    parser.add_argument("--wandb_project", type=str, default="qwen3-tts-entity-injection")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    gradient_accumulation_steps = args.gradient_accumulation_steps
    accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision="bf16", log_with="wandb")
    
    # Login to wandb if API key is provided
    WANDB_API_KEY = os.environ.get("WANDB_API_KEY")
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY)
    
    # Initialize wandb tracking
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config={
            "init_model_path": args.init_model_path,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_epochs": args.num_epochs,
            "speaker_name": args.speaker_name,
            "entity_loss_weight": args.entity_loss_weight,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision": "bf16",
            "pretrained_encoder": args.pretrained_encoder,
            "freeze_encoder": args.freeze_encoder,
        },
        init_kwargs={
            "wandb": {
                "name": args.wandb_run_name,
                "entity": args.wandb_entity,
            }
        }
    )

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)


    # Dataset paths - fill in with actual paths to HuggingFace datasets
    HF_datasets = {
        "hifi": "/speech/arjun/shoutrik/DATA/HiFi",        # TODO: Set actual path
        "GoogleTNLarge": "/speech/arjun/shoutrik/DATA/GoogleTNLarge",  # TODO: Set actual path
        "TextNormalisationSyntheticData": "/speech/arjun/shoutrik/DATA/TextNormalisationSyntheticData",          # TODO: Set actual path
    }

    dataset = TTSDataset(
        dataset_paths=HF_datasets, 
        processor=qwen3tts.processor, 
        config=config,
        codec_name="qwen3_12hz",
    )
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    # Initialize EntityInjectionModule with pretrained encoder
    # Options: "distilbert-base-uncased", "prajjwal1/bert-tiny", "prajjwal1/bert-mini", None (scratch)
    pretrained_encoder = args.pretrained_encoder if args.pretrained_encoder.lower() != "none" else None
    
    if accelerator.is_main_process:
        if pretrained_encoder:
            accelerator.print(f"Using pretrained encoder: {pretrained_encoder}")
            accelerator.print(f"  Freeze encoder: {args.freeze_encoder}")
        else:
            accelerator.print("Training entity encoder from scratch")
    
    entity_injection_config = EntityInjectionModuleConfig(
        hidden_size=config.talker_config.text_hidden_size,  # Qwen's text hidden dimension
        num_entities=len(dataset.entity_type_to_index),     # Number of entity types from dataset
        pretrained_encoder=pretrained_encoder,              # e.g., "distilbert-base-uncased" or None
        freeze_encoder=args.freeze_encoder,                 # Whether to freeze pretrained weights
        # Fallback settings if pretrained_encoder is None:
        num_layers=2,
        num_heads=8,
        dim_feedforward=config.talker_config.text_hidden_size * 4,
        dropout=0.1,
    )
    entity_injection_module = EntityInjectionModule(entity_injection_config)
    entity_injection_module = entity_injection_module.to(qwen3tts.model.dtype)
    
    # Special entity types that should be ignored in loss
    special_entities = dataset.special_types
    special_indices = torch.tensor([dataset.entity_type_to_index[t] for t in special_entities])

    model = Qwen3TTSModelWithEntityInjection(qwen3tts.model, config, entity_injection_module, special_indices).to(accelerator.device)
    for param in model.base_model.speaker_encoder.parameters():
        param.requires_grad = False

    optimizer, scheduler = get_optimizer_and_scheduler(model, args, args.num_epochs, len(train_dataloader), gradient_accumulation_steps)

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )


    # unwrapped = accelerator.unwrap_model(model)
    # model_gate = unwrapped.entity_injection_module.gate
    # optimizer_gate = optimizer.param_groups[2]['params'][0]  # Gate is in group 2

    # print(f"Model gate id: {id(model_gate)}")
    # print(f"Optimizer gate id: {id(optimizer_gate)}")
    # print(f"Same object: {id(model_gate) == id(optimizer_gate)}")
    # print(f"Model gate data_ptr: {model_gate.data_ptr()}")
    # print(f"Optimizer gate data_ptr: {optimizer_gate.data_ptr()}")

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        accelerator.print(f"Epoch {epoch} started with {len(train_dataloader)} batches...")
        step = 0
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(batch)
                target_speaker_embedding = outputs['target_speaker_embedding']
                # Total loss = TTS loss + sub-talker loss + weighted entity type loss
                loss = outputs['loss'] + outputs['sub_talker_loss'] + args.entity_loss_weight * outputs['entity_type_loss']

                accelerator.backward(loss)

                # if step % 20 == 0 and accelerator.sync_gradients and accelerator.is_main_process:
                #     unwrapped = accelerator.unwrap_model(model)
                #     if unwrapped.entity_injection_module.gate.grad is not None:
                #         accelerator.print(f"Gate grad (before zero): {unwrapped.entity_injection_module.gate.grad.item()}")

                # if accelerator.sync_gradients:
                #     unwrapped = accelerator.unwrap_model(model)
                #     base_model = unwrapped.base_model
                #     entity_module = unwrapped.entity_injection_module
                #     accelerator.clip_grad_norm_(base_model.parameters(), 2.0)
                #     # accelerator.clip_grad_norm_(entity_module.parameters(), 10.0)
                #     params_before = {name: p.clone() for name, p in entity_module.named_parameters()}

                optimizer.step()
                optimizer.zero_grad()
                            
                if accelerator.sync_gradients:
                    scheduler.step()
                    step += 1

                    # if step % 20 == 0:
                    #     all_updating, not_updating = check_parameter_changes(
                    #         accelerator, model, params_before, print_details=True
                    #     )
                    #     if not all_updating and accelerator.is_main_process:
                    #         accelerator.print(f"WARNING: {len(not_updating)} params not updating: {not_updating}")

                    if step % 20 == 0 and accelerator.is_main_process:
                        gamma_mean = outputs['gamma_mean'].item()
                        beta_mean = outputs['beta_mean'].item()
                        accelerator.print(
                            f"Epoch {epoch} | Step {step} | "
                            f"Loss: {loss.item():.4f} | "
                            f"TTS: {outputs['loss'].item():.4f} | "
                            f"Sub-talker: {outputs['sub_talker_loss'].item():.4f} | "
                            f"Entity: {outputs['entity_type_loss'].item():.4f} | "
                            f"γ: {gamma_mean:.4f} | β: {beta_mean:.4f}"
                        )
                        
                        global_step = int(((epoch * len(train_dataloader)) / gradient_accumulation_steps) + step)
                        accelerator.log(
                            {
                                "train/loss": loss.item(),
                                "train/tts_loss": outputs["loss"].item(),
                                "train/sub_talker_loss": outputs["sub_talker_loss"].item(),
                                "train/entity_type_loss": outputs["entity_type_loss"].item(),
                                "train/film_gamma_mean": gamma_mean,
                                "train/film_beta_mean": beta_mean,
                                "train/epoch": epoch,
                                "train/step": step,
                                "train/lr_entity": scheduler.get_last_lr()[1],
                            },
                            step=global_step,
                        )

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwraped_model = accelerator.unwrap_model(model)
            unwrapped_qwen3_tts_model = unwraped_model.base_model
            unwrapped_entity_injection_module = unwraped_model.entity_injection_module
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_qwen3_tts_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)
            
            # Save EntityInjectionModule separately
            entity_state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_entity_injection_module.state_dict().items()}
            entity_save_path = os.path.join(output_dir, "entity_injection_module.safetensors")
            save_file(entity_state_dict, entity_save_path)
            
            # Save entity type mapping for inference
            entity_type_mapping = {
                "entity_type_to_index": dataset.entity_type_to_index,
                "index_to_entity_type": dataset.index_to_entity_type,
                "entity_injection_config": {
                    "hidden_size": entity_injection_config.hidden_size,
                    "num_entities": entity_injection_config.num_entities,
                    "pretrained_encoder": entity_injection_config.pretrained_encoder,
                    "freeze_encoder": entity_injection_config.freeze_encoder,
                    "num_layers": entity_injection_config.num_layers,
                    "num_heads": entity_injection_config.num_heads,
                    "dim_feedforward": entity_injection_config.dim_feedforward,
                    "dropout": entity_injection_config.dropout,
                }
            }
            entity_mapping_path = os.path.join(output_dir, "entity_type_mapping.json")
            with open(entity_mapping_path, 'w') as f:
                json.dump(entity_type_mapping, f, indent=2)
            
            # Log checkpoint saved to wandb
            accelerator.log({"checkpoint/epoch": epoch}, step=epoch * len(train_dataloader))
    
    # End wandb tracking
    accelerator.end_training()

if __name__ == "__main__":
    train()
