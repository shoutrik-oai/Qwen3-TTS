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

import json
import os
from tqdm import tqdm
from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

JSONL_PATHS = [
    "/speech/arjun/shoutrik/DATA/giga/metadata.jsonl",
    "/speech/arjun/shoutrik/DATA/google_TN/metadata.jsonl",
    "/speech/arjun/shoutrik/DATA/hifi/metadata.jsonl",
    "/speech/arjun/shoutrik/DATA/libri_tts/metadata.jsonl"
]

tokenizer_model_path = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
output_jsonl_path = "/speech/arjun/shoutrik/DATA/qwen/metadata.jsonl"
device = "cuda:0"
write_every = 1000

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line.strip()) for line in f.readlines()]

def main():
    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        tokenizer_model_path,
        device_map=device,
    )

    total_lines = []
    for path in JSONL_PATHS:
        total_lines.extend(load_jsonl(path))

    # Handle case where output file doesn't exist yet
    already_done = set()
    if os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r') as f:
            lines = [json.loads(line.strip()) for line in f.readlines()]
            already_done = set([line['id'] for line in lines])

    lines_to_process = [line for line in total_lines if line['id'] not in already_done]

    with open(output_jsonl_path, 'a') as out_file:
        batch_lines = []
        batch_audios = []
        to_write = []
        
        for line in tqdm(lines_to_process, desc="Processing lines", total=len(lines_to_process)):
            batch_lines.append(line)
            batch_audios.append(line['wav_path'])

            if len(batch_lines) >= BATCH_INFER_NUM:
                enc_res = tokenizer_12hz.encode(batch_audios)
                for code, batch_line in zip(enc_res.audio_codes, batch_lines):
                    batch_line.pop("codes", None)  # Safe pop with default
                    batch_line['audio_codes'] = code.cpu().tolist()
                    wav_path = batch_line['wav_path']
                    batch_line.pop("wav_path", None)
                    batch_line["audio"] = wav_path
                    batch_line["language"] = "English"
                    batch_line["ref_audio"] = wav_path
                    to_write.append(batch_line)
                batch_lines.clear()
                batch_audios.clear()
            if len(to_write) >= write_every:
                for line in to_write:
                    out_file.write(json.dumps(line, ensure_ascii=False) + '\n')
                out_file.flush()
                to_write.clear()

        # Handle remaining items in the last batch
        if len(batch_audios) > 0:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, batch_line in zip(enc_res.audio_codes, batch_lines):
                batch_line.pop("codes", None)
                batch_line['audio_codes'] = code.cpu().tolist()
                wav_path = batch_line['wav_path']
                batch_line.pop("wav_path", None)
                batch_line["audio"] = wav_path
                batch_line["language"] = "English"
                batch_line["ref_audio"] = wav_path
                to_write.append(batch_line)

        # Flush any remaining items in to_write
        if len(to_write) > 0:
            for line in to_write:
                out_file.write(json.dumps(line, ensure_ascii=False) + '\n')
            out_file.flush()

if __name__ == "__main__":
    main()
