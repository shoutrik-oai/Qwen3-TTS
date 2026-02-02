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
    
    Input:
        text_embeddings: [B, T, D] - text token embeddings
        text_mask: [B, T] - 1 for valid text positions, 0 for padding
        
    Output:
        type_enriched: [B, T, D] - entity-enriched embeddings (residual connection)
        type_logits: [B, T, num_entities] - entity type predictions (if return_type_logits=True)
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
        self.gate = nn.Parameter(torch.zeros(1) + 0.1)  # Start small for pretrained compatibility
        self._init_weights()
    
    def _init_weights(self):
        # Initialize projection with small weights to start near identity
        nn.init.normal_(self.entity_delta_proj.weight, std=0.01)
        nn.init.zeros_(self.entity_delta_proj.bias)
        nn.init.xavier_uniform_(self.type_head.weight)
        nn.init.zeros_(self.type_head.bias)
    
    def forward(self, text_embeddings, text_mask=None, return_type_logits=False):
        """
        Args:
            text_embeddings: [B, T, D] text token embeddings
            text_mask: [B, T] attention mask (1 = valid, 0 = padding)
            return_type_logits: whether to return entity type logits for supervision
            
        Returns:
            type_enriched: [B, T, D] entity-enriched embeddings
            type_logits: [B, T, num_entities] (optional) entity type predictions
        """
        # Save input dtype for later casting

        
        # TransformerEncoder expects src_key_padding_mask where True = ignore
        if text_mask is not None:
            src_key_padding_mask = (text_mask == 0)
        else:
            src_key_padding_mask = None
        
        # Encode with entity-aware transformer
        type_hidden = self.entity_encoder(text_embeddings, src_key_padding_mask=src_key_padding_mask)
        
        # Project and gate the residual - cast intermediate results to input dtype
        type_delta = self.entity_delta_proj(type_hidden)
        type_enriched = text_embeddings + self.gate * type_delta
        
        if return_type_logits:
            type_logits = self.type_head(type_enriched)
            return type_enriched, type_logits
        
        return type_enriched


class Qwen3TTSModelWithEntityInjection(nn.Module):
    def __init__(self, base_model, qwen3_config, entity_injection_module, entity_loss_weights):
        super().__init__()
        self.base_model = base_model
        self.qwen3_config = qwen3_config
        self.entity_injection_module = entity_injection_module
        self.register_buffer('entity_loss_weights', entity_loss_weights)
        
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

        input_text_embedding = self.base_model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
        text_only_mask = batch['text_only_mask']  
        input_text_embedding, entity_type_logits = self.entity_injection_module(
            input_text_embedding, 
            text_mask=text_only_mask.long(),
            return_type_logits=True
        )
        input_text_embedding = input_text_embedding.to(torch.bfloat16)
        input_text_embedding = self.base_model.talker.text_projection(input_text_embedding)

        entity_labels = entities[:, :, 0]  # [B, T]
        
        entity_type_loss = F.cross_entropy(
            entity_type_logits.view(-1, entity_type_logits.size(-1)),
            entity_labels.view(-1),
            weight=self.entity_loss_weights,
            ignore_index=-100,  # Ignore non-entity positions
        )
    
        if torch.isnan(entity_type_loss):
            entity_type_loss = 0.0 * entity_type_logits.sum()
        
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
        gate_value = self.entity_injection_module.gate

        return {
            'loss': outputs.loss,
            'sub_talker_loss': sub_talker_loss,
            'entity_type_loss': entity_type_loss,
            "gate_value": gate_value,
            "target_speaker_embedding": target_speaker_embedding,
        }

def get_optimizer_and_scheduler(model, args, num_epochs, num_batches, accum_grad):
    base_model = model.base_model
    entity_injection_module = model.entity_injection_module

    entity_params_no_gate = [p for n, p in entity_injection_module.named_parameters() if "gate" not in n]
    gate_param = [entity_injection_module.gate]

    entity_lr_multiplier = 10
    entity_initial_lr = args.lr * entity_lr_multiplier

    optimizer_groups = [
        {
            "params": list(base_model.parameters()),
            "lr": args.lr,
            "weight_decay": 0.01,
            "name": "base_model",
        },
        {
            "params": entity_params_no_gate,
            "lr": entity_initial_lr,
            "weight_decay": 0.01,
            "name": "entity_injection_module_no_gate",
        },
        {
            "params": gate_param,
            "lr": entity_initial_lr,
            "weight_decay": 0.0,
            "name": "entity_injection_module_gate",
        },
    ]
    optimizer = AdamW(optimizer_groups)

    total_steps = int((num_epochs * num_batches) / accum_grad)
    warmup_steps = int(total_steps * 0.05)

    def lr_lambda_base_model(current_step: int):
        return 1.0

    def lr_lambda_entity(current_step):
        if current_step < warmup_steps:
            ratio = 0.1 + ((1-0.1) / warmup_steps) * current_step
            return ratio
        else:
            decay_steps = total_steps - warmup_steps
            steps_after_warmup = current_step - warmup_steps
            target_ratio = args.lr / entity_initial_lr 
            progress = float(steps_after_warmup) / float(max(1, decay_steps))
            return target_ratio ** progress
    
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[
            lr_lambda_base_model,
            lr_lambda_entity,
            lr_lambda_entity,
        ]
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
        attn_implementation="sdpa",
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

    # Initialize EntityInjectionModule
    entity_injection_config = EntityInjectionModuleConfig(
        hidden_size=config.talker_config.text_hidden_size,  # Match text embedding dimension
        num_entities=len(dataset.entity_type_to_index),     # Number of entity types from dataset
        num_layers=2,
        num_heads=8,
        dim_feedforward=config.talker_config.text_hidden_size * 4,
        dropout=0.1,
        activation="gelu",
        batch_first=True,
    )
    entity_injection_module = EntityInjectionModule(entity_injection_config)
    entity_injection_module = entity_injection_module.to(qwen3tts.model.dtype)
    
    special_entities = dataset.special_types
    special_indices = torch.tensor([dataset.entity_type_to_index[t] for t in special_entities])
    entity_loss_weights = torch.ones(len(dataset.entity_type_to_index))
    entity_loss_weights[special_indices] = 0.1
    entity_loss_weights = entity_loss_weights.to(torch.bfloat16).to(accelerator.device)

    model = Qwen3TTSModelWithEntityInjection(qwen3tts.model, config, entity_injection_module, entity_loss_weights).to(accelerator.device)
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
                        current_gate = accelerator.unwrap_model(model).entity_injection_module.gate.item()
                        accelerator.print(
                            f"Epoch {epoch} | Step {step} | "
                            f"Loss: {loss.item():.4f} | "
                            f"TTS: {outputs['loss'].item():.4f} | "
                            f"Sub-talker: {outputs['sub_talker_loss'].item():.4f} | "
                            f"Entity: {outputs['entity_type_loss'].item():.4f} | "
                            f"Gate: {current_gate:.6f}"
                        )
                        
                        global_step = int(((epoch * len(train_dataloader)) / gradient_accumulation_steps) + step)
                        accelerator.log(
                            {
                                "train/loss": loss.item(),
                                "train/tts_loss": outputs["loss"].item(),
                                "train/sub_talker_loss": outputs["sub_talker_loss"].item(),
                                "train/entity_type_loss": outputs["entity_type_loss"].item(),
                                "train/entity_gate": current_gate,
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
                    "num_layers": entity_injection_config.num_layers,
                    "num_heads": entity_injection_config.num_heads,
                    "dim_feedforward": entity_injection_config.dim_feedforward,
                    "dropout": entity_injection_config.dropout,
                    "activation": entity_injection_config.activation,
                    "batch_first": entity_injection_config.batch_first,
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
