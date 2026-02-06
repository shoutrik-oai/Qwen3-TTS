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
import math
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
from dataset import TTSDataset, DistributedMixedSourceBatchSampler
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig
from dataclasses import dataclass
from torch.optim.lr_scheduler import LambdaLR
from accelerate.utils import DistributedDataParallelKwargs
from transformers import AutoModel, AutoConfig as HFAutoConfig
# from peft import LoraConfig, get_peft_model


@dataclass
class EntityInjectionModuleConfig:
    hidden_size: int  # Qwen's text hidden size
    num_entities: int
    # Pretrained encoder settings
    num_layers: int = 2
    num_heads: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    entity_prob: float = 0.0


class FiLMLayer(nn.Module):
    def __init__(self, hidden_size, num_entities):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_entities = num_entities
        
        self.entity_type_embeddings = nn.Embedding(num_entities, hidden_size)
        self.film_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size * 2),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.entity_type_embeddings.weight, std=0.02)
        for module in self.film_mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                nn.init.zeros_(module.bias)
        
        # Make final layer output very small values initially
        final_layer = self.film_mlp[-1]
        nn.init.normal_(final_layer.weight, std=0.001)
        nn.init.zeros_(final_layer.bias)

    def forward(self, hidden_states, entity_logits, temperature=1.0):
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


class EntityInjectionModule(nn.Module):
    def __init__(self, config: EntityInjectionModuleConfig):
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
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.entity_head.weight)
        nn.init.zeros_(self.entity_head.bias)
        nn.init.xavier_uniform_(self.entity_detector.weight)
        prior_prob = max(self.config.entity_prob, 1e-6)  # Clamp to avoid div by zero
        bias_init = math.log(prior_prob / (1 - prior_prob))
        nn.init.constant_(self.entity_detector.bias, bias_init)
    
    def forward(self, text_embeddings, text_mask=None, return_details=False):
        if text_mask is not None:
            src_key_padding_mask = (text_mask == 0)
        else:
            src_key_padding_mask = None

        text_embeddings = self.positional_embedding(text_embeddings)
        entity_hidden = self.entity_encoder(text_embeddings, src_key_padding_mask=src_key_padding_mask)

        entity_logits = self.entity_head(entity_hidden) 
        entity_detection_logits = self.entity_detector(entity_hidden)
        is_entity = torch.sigmoid(entity_detection_logits)

        conditioned, gamma, beta = self.film_layer(text_embeddings, entity_logits)

        conditioned = text_embeddings * (1 - is_entity) + conditioned * is_entity
        
        if return_details:
            return conditioned, entity_logits, entity_detection_logits, gamma, beta
        
        return conditioned, entity_logits, entity_detection_logits


class Qwen3TTSModelWithEntityInjection(nn.Module):
    def __init__(self, base_model, qwen3_config, entity_injection_module, special_indices, special_index, entity_prob):
        super().__init__()
        self.base_model = base_model
        self.qwen3_config = qwen3_config
        self.entity_injection_module = entity_injection_module
        self.special_indices = special_indices
        self.special_index = special_index
        self.entity_prob = entity_prob
        
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
        
        conditioned_embedding, entity_logits, entity_detection_logits, gamma, beta = self.entity_injection_module(
            input_text_embedding, 
            text_mask=text_only_mask.long(),
            return_details=True
        )
        
        conditioned_embedding = conditioned_embedding.to(torch.bfloat16)
        input_text_embedding = self.base_model.talker.text_projection(conditioned_embedding)

        entity_labels = entities[:, :, 0].clone()  # [B, T]

        positions_valid_for_entity_detection = (entity_labels != -100) & text_only_mask
        
        special_mask = torch.isin(entity_labels, self.special_indices.to(entity_labels.device))
        is_entity_labels = ~special_mask

        entity_labels_for_type = entity_labels.clone()
        entity_labels_for_type[special_mask] = -100
        
        entity_type_loss = F.cross_entropy(
            entity_logits.view(-1, entity_logits.size(-1)),
            entity_labels_for_type.view(-1),
            ignore_index=-100,
        )
        pos_weight = torch.tensor((1 - self.entity_prob) / self.entity_prob, device=entity_detection_logits.device)
        entity_detection_loss = F.binary_cross_entropy_with_logits(
            entity_detection_logits[positions_valid_for_entity_detection].view(-1),
            is_entity_labels[positions_valid_for_entity_detection].float().view(-1),
            pos_weight=pos_weight,
        )
        
        if torch.isnan(entity_type_loss):
            entity_type_loss = 0.0 * entity_logits.sum() + 0.0 * gamma.sum() + 0.0 * beta.sum()
        else:
            entity_type_loss = entity_type_loss + 0.0 * gamma.sum() + 0.0 * beta.sum()
        
        entity_loss = entity_type_loss + entity_detection_loss
        
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
            'entity_detection_loss': entity_detection_loss,
            'entity_loss': entity_loss,
            'gamma_mean': gamma_mean,
            'beta_mean': beta_mean,
            'target_speaker_embedding': target_speaker_embedding,
        }

def get_optimizer_and_scheduler(model, args, num_epochs, num_batches, accum_grad):
    base_model = model.base_model
    entity_module = model.entity_injection_module

    # Categorize parameters
    projection_params = []
    entity_head_params = []
    entity_detector_params = []
    film_layer_params = []
    scratch_encoder_params = []
    
    for name, param in entity_module.named_parameters():
        if not param.requires_grad:
            continue
        elif "entity_head" in name:
            entity_head_params.append(param)
        elif "entity_detector" in name:
            entity_detector_params.append(param)
        elif "film_layer" in name:
            film_layer_params.append(param)
        elif "entity_encoder" in name:
            scratch_encoder_params.append(param)

    # Learning rates
    base_lr = args.lr
    new_module_lr = args.lr * 100    # Higher LR for new modules (train from scratch)

    optimizer_groups = [
        {
            "params": list(base_model.parameters()),
            "lr": base_lr,
            "weight_decay": 0.01,
            "name": "base_tts_model",
        },
    ]
    
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
    
    if entity_head_params:
        optimizer_groups.append({
            "params": entity_head_params,
            "lr": new_module_lr,
            "weight_decay": 0.01,
            "name": "entity_head",
        })
    
    if entity_detector_params:
        optimizer_groups.append({
            "params": entity_detector_params,
            "lr": new_module_lr,
            "weight_decay": 0.01,
            "name": "entity_detector",
        })
    
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
        "hifi": "/speech/arjun/shoutrik/DATA/HF_datasets/HiFi",        # TODO: Set actual path
        "GoogleTNLarge": "/speech/arjun/shoutrik/DATA/HF_datasets/GoogleTN_V1",  # TODO: Set actual path
        "GoogleTNLarge_v2": "/speech/arjun/shoutrik/DATA/HF_datasets/GoogleTN_V2",
        "TextNormalisationSyntheticData": "/speech/arjun/shoutrik/DATA/HF_datasets/TextNormOnlyAllTypes",
        "TextNormalisationSyntheticDataOnlyNumbers": "/speech/arjun/shoutrik/DATA/HF_datasets/TextNormOnlyNumbers",          # TODO: Set actual path
    }

    dataset = TTSDataset(
        dataset_paths=HF_datasets, 
        processor=qwen3tts.processor, 
        config=config,
        codec_name="qwen3_12hz",
    )
    sampler = DistributedMixedSourceBatchSampler(dataset, args.batch_size, num_replicas=accelerator.num_processes, rank=accelerator.process_index, shuffle=True)
    train_dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=dataset.collate_fn)

    n_entities = dataset.n_entities
    n_tokens = dataset.n_tokens
    entity_prob = n_entities / n_tokens
    
    entity_injection_config = EntityInjectionModuleConfig(
        hidden_size=config.talker_config.text_hidden_size,  # Qwen's text hidden dimension
        num_entities=len(dataset.entity_type_to_index),     # Number of entity types from dataset          # e.g., "distilbert-base-uncased" or None              # Whether to freeze pretrained weights
        num_layers=2,
        num_heads=8,
        dim_feedforward=config.talker_config.text_hidden_size * 4,
        dropout=0.1,
        entity_prob=entity_prob,
    )
    entity_injection_module = EntityInjectionModule(entity_injection_config)
    entity_injection_module = entity_injection_module.to(qwen3tts.model.dtype)
    
    special_entities = dataset.special_types
    special_indices = torch.tensor([dataset.entity_type_to_index[t] for t in special_entities])
    special_index = dataset.entity_type_to_index["SPECIAL"]

    model = Qwen3TTSModelWithEntityInjection(qwen3tts.model, config, entity_injection_module, special_indices, special_index, entity_prob).to(accelerator.device)
    for param in model.base_model.speaker_encoder.parameters():
        param.requires_grad = False

    optimizer, scheduler = get_optimizer_and_scheduler(model, args, args.num_epochs, len(train_dataloader), gradient_accumulation_steps)

    model, optimizer, scheduler, train_dataloader = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()
    step = 0
    for epoch in range(num_epochs):
        accelerator.print(f"Epoch {epoch} started with {len(train_dataloader)} batches...")
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(batch)
                target_speaker_embedding = outputs['target_speaker_embedding']
                # Total loss = TTS loss + sub-talker loss + weighted entity type loss
                loss = outputs['loss'] + outputs['sub_talker_loss'] + args.entity_loss_weight * outputs['entity_loss']

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
                            f"EntityType: {outputs['entity_type_loss'].item():.4f} | "
                            f"EntityDet: {outputs['entity_detection_loss'].item():.4f} | "
                            f"γ: {gamma_mean:.4f} | β: {beta_mean:.4f}"
                        )
                        
                        accelerator.log(
                            {
                                "train/loss": loss.item(),
                                "train/tts_loss": outputs["loss"].item(),
                                "train/sub_talker_loss": outputs["sub_talker_loss"].item(),
                                "train/entity_type_loss": outputs["entity_type_loss"].item(),
                                "train/entity_detection_loss": outputs["entity_detection_loss"].item(),
                                "train/entity_loss": outputs["entity_loss"].item(),
                                "train/film_gamma_mean": gamma_mean,
                                "train/film_beta_mean": beta_mean,
                                "train/epoch": epoch,
                                "train/step": step,
                                "train/lr_entity": scheduler.get_last_lr()[1],
                            },
                            step=step,
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
            accelerator.log({"checkpoint/epoch": epoch}, step=step)
    
    # End wandb tracking
    accelerator.end_training()

if __name__ == "__main__":
    train()
