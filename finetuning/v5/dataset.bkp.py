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
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import json
from transformers import AutoConfig

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config:Qwen3TTSConfig, lag_num = -1):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.special_types = ["SPECIAL", "PLAIN_WORD"]

        assert self.processor.tokenizer.is_fast, "Processor must be a fast tokenizer"

        # Build entity type vocabulary with required base types
        entity_types = set([entity["type"] for line in data_list for entity in line["entities"]])
        for special_type in self.special_types:
            if special_type not in entity_types:
                entity_types.add(special_type)
        entity_types = sorted(list(entity_types))
        self.entity_type_to_index = {entity_type: i for i, entity_type in enumerate(entity_types)}
        self.index_to_entity_type = {v: k for k, v in self.entity_type_to_index.items()}

    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        
        audio, sr = librosa.load(x, sr=None, mono=True)
        if sr is not None and sr != 24000:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=24000)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), 24000

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                audio, sr = self._load_audio_to_np(a)
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                audio, sr = a[0].astype(np.float32), int(a[1])
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
            
            # Normalize waveform to [-1, 1] range
            max_val = np.abs(audio).max()
            if max_val > 1.0:
                audio = audio / max_val
            
            out.append((audio, sr))
        return out

    
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True, return_offsets_mapping=True)
        offsets = input.pop("offset_mapping", None)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id, offsets
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        return mels
    
    def _align_entities_with_text(self, offset_mapping, entities):
        num_tokens = offset_mapping.shape[0]
        token_labels = ["PLAIN_WORD"] * num_tokens

        prefix_len = len("<|im_start|>assistant\n")

        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == token_end == 0:
                token_labels[idx] = "SPECIAL"
                continue
            
            for entity in entities:
                ent_start = entity['start'] + prefix_len
                ent_end = entity['end'] + prefix_len
                ent_type = entity['type']
                
                if token_start < ent_end and token_end > ent_start:
                    token_labels[idx] = ent_type
                    break

        plain_idx = self.entity_type_to_index.get("PLAIN_WORD")
        token_label_ids = [self.entity_type_to_index.get(label, plain_idx) for label in token_labels]
        return token_labels, token_label_ids

    def __getitem__(self, idx):
        item = self.data_list[idx]

        audio_path  = item["audio"]
        text        = item["text"]
        audio_codes = item["audio_codes"]
        language        = item.get('language','Auto')
        ref_audio_path  = item['ref_audio']
        entities        = item['entities']

        # print(f"text: {text}")
        text = self._build_assistant_text(text)
        # print(f"text after build: {text}")

        text_ids, offsets = self._tokenize_texts(text)
        text_ids = text_ids[:,:-5]
        offsets = offsets[0, 3:-5] if offsets.dim() == 3 else offsets[3:-5]
        entity_labels, entity_label_ids = self._align_entities_with_text(offsets, entities)

        # for text_id, entity_label, entity_label_id in zip(text_ids[0, 3:], entity_labels, entity_label_ids):
        #     print(f"text id : {text_id}, text token: {self.processor.tokenizer.decode(text_id)}, entity label: {entity_label} ({entity_label_id})\n")


        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav,sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)
        # print(f"length of text : {len(text_ids[0, 3:])}")
        # print(f"length of entity labels : {len(entity_label_ids)}")
        # print(f"text : {self.processor.tokenizer.decode(text_ids[0, 3:])}")
        # print(f"entity labels : {entity_labels}")

        return {
            "text_ids":text_ids,    # 1 , t
            "audio_codes":audio_codes,      # t, 16
            "ref_mel":ref_mel,
            "entities":entity_label_ids
        }
       
    def collate_fn(self, batch):

        # print(batch)

        assert self.lag_num == -1

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        max_length = max(item_length) + 8
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)

        entities = torch.full((b, t, 1), -100, dtype=torch.long)
        text_only_mask = torch.zeros((b, t), dtype=torch.bool)  # Mask for text positions only (not audio)

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]
            
            # text channel
            input_ids[i,  :3, 0] = text_ids[0,:3]
            input_ids[i, 3:7, 0] = self.config.tts_pad_token_id
            input_ids[i,   7, 0] = self.config.tts_bos_token_id
            input_ids[i, 8:8+text_ids_len-3, 0] = text_ids[0,3:]

            # print(f"text ids length : {text_ids_len}")
            # print(f"text ids : {input_ids}")

            # Only set entity labels for actual entities (not SPECIAL or PLAIN_WORD)
            # All other positions remain -100
            entity_labels = torch.tensor(data['entities'], dtype=torch.long)
            # special_indices = torch.tensor([self.entity_type_to_index[t] for t in self.special_types])
            # is_actual_entity = ~torch.isin(entity_labels, special_indices)
            entity_len = len(entity_labels)
            # print(f"entity labels length : {entity_len}")

            # entities[i, 8:8+entity_len, 0] = torch.where(is_actual_entity, entity_labels, -100)
            entities[i, 8:8+entity_len, 0] = entity_labels
            # print(f"entities : {entities}")

            input_ids[i,   8+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, 8+text_ids_len-2:8+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id
            text_embedding_mask[i,  :8+text_ids_len+codec_ids_len] = True
            
            # text_only_mask: covers ONLY actual text content tokens (position 8 onwards)
            # Excludes: header (0-7), EOS, and audio positions
            # This aligns exactly with where entity labels are defined
            text_only_mask[i, 8:8+text_ids_len-3] = True  # text_ids_len-3 = remaining tokens after first 3
            # print(f"text only mask : {text_only_mask}")
            # codec channel
            # input_ids[i,   :3, 1] = 0
            input_ids[i,    3:8 ,1] = torch.tensor(
                                        [
                                            self.config.talker_config.codec_nothink_id,
                                            self.config.talker_config.codec_think_bos_id,
                                            self.config.talker_config.codec_think_eos_id,
                                            0,     # for speaker embedding
                                            self.config.talker_config.codec_pad_id       
                                        ]
                                    )
            input_ids[i,    8:8+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    8+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    8+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i,    8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    8+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, 8+text_ids_len-1:8+text_ids_len-1+codec_ids_len,:] = audio_codecs

            codec_embedding_mask[i, 3:8+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 6] = False       # for speaker embedding

            codec_mask[i,   8+text_ids_len-1:8+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :8+text_ids_len+codec_ids_len] = True
        
        ref_mels = [data['ref_mel'] for data in batch]  # List of [1, T_i, 128]
        max_mel_len = max(m.size(1) for m in ref_mels)
        ref_mels = torch.cat([
            F.pad(m, (0, 0, 0, max_mel_len - m.size(1))) 
            for m in ref_mels
        ], dim=0) 

        return {
            'input_ids':input_ids,
            'ref_mels':ref_mels,
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask,
            'entities':entities,
            'text_only_mask':text_only_mask,  # [B, T] mask for text positions only (for EntityInjectionModule)
        }