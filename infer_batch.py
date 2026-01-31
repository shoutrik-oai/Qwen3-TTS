import os
import json
import random
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from tqdm import tqdm

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

jsonl_path = "/speech/arjun/shoutrik/DATA/google_TN_large/prepared_data.jsonl"
out_file_path = "/speech/arjun/shoutrik/DATA/google_TN_large/generated_metadata.jsonl"
AUDIO_DIR = "/speech/arjun/shoutrik/DATA/google_TN_large/AUDIO"  # TODO: set this
batch_size = 16
random_drop = 0.4
speaker_choices = ["Ryan", "Aiden"]  # TODO: populate with speaker options
duration = 0
duration_limit = 1000 * 3600

def process_batch(batch_text, batch_ids, batch_written_text, batch_entities, batch_speaker, f_out):
    """Process a batch of texts and write outputs."""
    if not batch_text:
        return
    language = ["English"] * len(batch_text)
    wavs, sr = model.generate_custom_voice(
        text=batch_text,
        language=language,
        speaker=batch_speaker,
    )
    for i, wav in enumerate(wavs):
        sf.write(os.path.join(AUDIO_DIR, batch_ids[i] + ".wav"), wav, sr)
        duration += len(wav) / sr
        print("duration: ", duration)
        f_out.write(json.dumps({
            "id": batch_ids[i],
            "written": batch_written_text[i],
            "spoken": batch_text[i],
            "entities": batch_entities[i],
            "speaker": batch_speaker[i],
            "duration": duration
        }) + "\n")
    f_out.flush()


with open(jsonl_path, "r") as f, open(out_file_path, "w") as f_out:
    lines = f.readlines()
    batch_text = []
    batch_ids = []
    batch_written_text = []
    batch_entities = []
    batch_speaker = []
    
    for line in tqdm(lines, desc="Processing lines", total=len(lines)):
        data = json.loads(line)
        id_ = data["id"]
        text = data["spoken"]
        entities = data["entities"]
        written_text = data["written"]
        
        if len(entities) == 0 and random.random() < random_drop:
            continue
        
        wav_path = os.path.join(AUDIO_DIR, id_ + ".wav")
        if os.path.exists(wav_path):
            continue
        
        batch_text.append(text)
        batch_ids.append(id_)
        batch_entities.append(entities)
        batch_written_text.append(written_text)
        speaker = random.choice(speaker_choices)
        batch_speaker.append(speaker)
        
        if len(batch_text) == batch_size:
            process_batch(batch_text, batch_ids, batch_written_text, batch_entities, batch_speaker, f_out)
            batch_text = []
            batch_ids = []
            batch_entities = []
            batch_written_text = []
            batch_speaker = []
        
        if duration > duration_limit:
            break
    
    # Process remaining items in the last partial batch
    process_batch(batch_text, batch_ids, batch_written_text, batch_entities, batch_speaker, f_out)
