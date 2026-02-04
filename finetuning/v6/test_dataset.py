import os
import sys
import tempfile
import numpy as np
import torch
from datasets import Dataset

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from transformers import AutoConfig

from test_alignment import create_mock_dataset
from dataset import TTSDataset


MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

print(f"\nLoading model from: {MODEL_PATH}")

# Load model and processor
qwen3tts = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
)

processor = qwen3tts.processor
config = qwen3tts.model.config

tts_dataset = TTSDataset(
    dataset_paths={
        "test": "/speech/arjun/shoutrik/DATA/TextNormalisationSyntheticData",
    },
    processor=processor,
    config=config,
    codec_name="qwen3_12hz",
    lag_num=-1,
)

for sample in tts_dataset:
    print(f"id : {sample['id']}")
    print(f"text_ids : {sample['text_ids'][0].tolist()}")
    print(f"entities : {sample['entities']}")
    print(f"text_detokenized :")
    for i, text_id in enumerate(sample['text_ids'][0].tolist()[3:]):
        print(f"{i} : {processor.tokenizer.decode(text_id)} | {sample['entities'][i]}", end="\n")
    print("\n\n")