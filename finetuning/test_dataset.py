import torch
from transformers import AutoConfig
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.utils.data import DataLoader
from dataset import TTSDataset
import json


data_path = "/speech/arjun/shoutrik/Qwen3-TTS/finetuning/small_test.jsonl" 
MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

qwen3tts = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
config = AutoConfig.from_pretrained(MODEL_PATH)

with open(data_path, 'r') as f:
    data = [json.loads(line) for line in f]

dataset = TTSDataset(data, qwen3tts.processor, config)

dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn, num_workers=1)
for batch in dataloader:
    print(batch)
    for k, v in batch.items():
        print(k, v.shape)
    break