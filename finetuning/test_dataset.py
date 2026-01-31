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


# Create test data with entities
test_data = [
    {
        "audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "text": "Dr. Smith paid $500 on January 15th.",
        "audio_codes": [[i % 1024 for _ in range(16)] for i in range(50)],  # 50 frames
        "language": "English",
        "ref_audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "entities": [
            {"type": "TITLE", "start": 0, "end": 3},      # "Dr."
            {"type": "NAME", "start": 4, "end": 9},       # "Smith"
            {"type": "CURRENCY", "start": 15, "end": 19}, # "$500"
            {"type": "DATE", "start": 23, "end": 35},     # "January 15th"
        ]
    },
    {
        "audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "text": "Call 555-1234 now.",
        "audio_codes": [[i % 1024 for _ in range(16)] for i in range(30)],  # 30 frames
        "language": "English",
        "ref_audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "entities": [
            {"type": "PHONE", "start": 5, "end": 13},  # "555-1234"
        ]
    },
    {
        "audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "text": "The price is $1,234.56 for item ABC-123.",
        "audio_codes": [[i % 1024 for _ in range(16)] for i in range(60)],  # 60 frames
        "language": "English",
        "ref_audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "entities": [
            {"type": "CURRENCY", "start": 13, "end": 22},  # "$1,234.56"
            {"type": "CODE", "start": 32, "end": 39},      # "ABC-123"
        ]
    },
    {
        "audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "text": "Meeting at 3:30 PM.",
        "audio_codes": [[i % 1024 for _ in range(16)] for i in range(25)],  # 25 frames
        "language": "English",
        "ref_audio": "/speech/arjun/shoutrik/DATA/giga/SEGMENTED_AUDIO_22k/AUD0000000004_S0000001.wav",
        "entities": [
            {"type": "TIME", "start": 11, "end": 18},  # "3:30 PM"
        ]
    },
]


dataset = TTSDataset(test_data, qwen3tts.processor, config)
sample = dataset[0]
sample_after_collate = dataset.collate_fn([sample])
# print(sample)

# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn, num_workers=1)
# for batch in dataloader: