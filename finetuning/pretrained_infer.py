import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
# Reference audio for cloning
ref_audio = "/speech/Database/hi_fi_tts_v0/audio/12787_other/13887/dayoffate_09_roe_0035.flac"
ref_text  = "i don't like to do anything in a hurry least of all to eat my dinner"

# Generate speech
wavs, sr = model.generate_voice_clone(
    text="The company reported $2.5 million in revenue for Q3 2024, up 15% from last year. The CEO noted that their 500 GB storage plan now costs only $9.99 per month.",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)

sf.write("pretrained_money.wav", wavs[0], sr)