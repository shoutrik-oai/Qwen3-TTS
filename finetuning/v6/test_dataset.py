import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets

from dataset import TTSDataset
from qwen_tts import Qwen3TTSModel


HF_datasets = {
    "TextNormalisationSyntheticData": "/speech/arjun/shoutrik/DATA/HF_datasets/TextNormOnlyAllTypes",
}

MODEL_PATH = "/speech/arjun/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/5d83992436eae1d760afd27aff78a71d676296fc"

qwen3tts = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
config = AutoConfig.from_pretrained(MODEL_PATH)

dataset = TTSDataset(
    dataset_paths=HF_datasets, 
    processor=qwen3tts.processor, 
    config=config,
    codec_name="qwen3_12hz",
)

train_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)


for batch in train_dataloader:
    input_ids = batch["input_ids"]
    entities = batch["entities"]
    text_only_mask = batch["text_only_mask"]

    input_text_ids = input_ids[:, :, 0]
    entity_labels = entities[:, :, 0]

    entity_labels_after_special_mask = entity_labels.clone()

    special_indices = torch.tensor([dataset.entity_type_to_index[t] for t in dataset.special_types])
    special_mask = torch.isin(entity_labels, special_indices)
    entity_labels_after_special_mask[special_mask] = -100

    print(f"shape of input_text_ids: {input_text_ids.shape}")
    print(f"shape of entities: {entity_labels.shape}")

    for t, e, t_only_mask, e_after_special_mask in zip(input_text_ids[0], entity_labels[0], text_only_mask[0], entity_labels_after_special_mask[0]):
        t = qwen3tts.processor.tokenizer.decode(t)
        print(f"Text: {t}\tEntity : {e}\tEntity Label: {dataset.index_to_entity_type.get(int(e), 'IGNORED')}\tEntity Label after special mask: {dataset.index_to_entity_type.get(int(e_after_special_mask), 'IGNORED')}\ttext_only_mask: {t_only_mask}\n")

    print("-"*100)
    print("-"*100)



