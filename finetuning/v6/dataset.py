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
from typing import Any, List, Tuple, Union, Optional
import os

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
from tqdm import tqdm
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import Sampler

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    def __init__(
        self, 
        dataset_paths: dict, 
        processor, 
        config: Qwen3TTSConfig, 
        codec_name: str = "qwen3_12hz",
        lag_num: int = -1
    ):
        """
        Args:
            dataset_paths: Dict mapping dataset name to HuggingFace dataset path
                          e.g. {"hifi": "/path/to/hifi_dataset", "google_tn": "/path/to/google_tn"}
            processor: Qwen3TTS processor
            config: Qwen3TTSConfig
            codec_name: Name of the codec (used to find audio_codes_{codec_name} column)
            lag_num: Lag number for training
        """
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.codec_column = f"audio_codes_{codec_name}"
        self.special_types = ["SPECIAL", "PLAIN_WORD", "PLAIN"]
        self.codec_name = codec_name

        assert self.processor.tokenizer.is_fast, "Processor must be a fast tokenizer"

        self._load_dataset(dataset_paths)

        self._filter_dataset()
        self._expand_dataset()

        self._prepare_entities_map_and_speaker_map()
        self.data_list = self.data_list.map(self._prepare_text_and_entities, batched=True, desc="Preparing text and entities...")
        self.data_list = self.data_list.remove_columns(["text", "entities"])


        print(f"Data prepared successfully !!")
        print(f"Total samples: {len(self.data_list)}")
        print(f"Total entity types: {len(self.entity_type_to_index)}")
        print(f"Total speakers: {len(self.speaker_map.items())}")
        print("Entites : ", self.entity_type_to_index)
        print("Speakers : ", self.speaker_map.items())
        print(f"Total duration: {sum(self.data_list['duration']) / 3600} hours")

        self.clean_samples = [idx for idx in range(len(self.data_list)) if self.data_list['dataset_name'][idx] == "hifi"]
        self.spoken_samples = [idx for idx in range(len(self.data_list)) if self.data_list["id"][idx].endswith("_spoken") and self.data_list['dataset_name'][idx] != "hifi"]
        self.written_samples = [idx for idx in range(len(self.data_list)) if self.data_list["id"][idx].endswith("_written") and self.data_list['dataset_name'][idx] != "hifi"]

        self.n_tokens = sum(len(data['text_ids'][0]) for data in self.data_list)
        self.n_entities = sum([sum(1 for id_ in data["entity_label_ids"] if self.index_to_entity_type[id_] not in self.special_types) for data in self.data_list])
        print(f"Total tokens: {self.n_tokens}")
        print(f"Total entities: {self.n_entities}")

    def _load_dataset(self, dataset_paths):
        print("Loading datasets...")
        datasets_list = []
        for dataset_name, dataset_path in dataset_paths.items():
            print(f"  Loading {dataset_name} from {dataset_path}...")
            dataset = load_from_disk(dataset_path)
            dataset = dataset.map(
                lambda x: {"reference_audio_path": os.path.join(dataset_path, x['reference_audio_path'])},
                desc=f"Adding root to {dataset_name}"
            )
            dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
            datasets_list.append(dataset)
            print(f"    Loaded {len(dataset)} samples")
        
        self.data_list = concatenate_datasets(datasets_list)
        print(f"Total samples after concatenation: {len(self.data_list)}")
        self.data_list = self.data_list.shuffle(seed=42)
        self.samples_per_dataset = {}


    def _filter_dataset(self):

        duration_before_filter = sum(self.data_list['audio_duration']) / 3600
        self.data_list = self.data_list.filter(lambda x: x['audio_duration'] > 2 and x['audio_duration'] < 30)
        duration_after_filter = sum(self.data_list['audio_duration']) / 3600
        print(f"Total samples after filtering for duration between 2 and 30 seconds: {len(self.data_list)}")
        print(f"Duration before filter: {duration_before_filter} hours")
        print(f"Duration after filter: {duration_after_filter} hours")
        print(f"Filtering percentage: {((duration_before_filter - duration_after_filter) / duration_before_filter) * 100}%")

    def _expand_dataset(self):
        self.data_list = self.data_list.map(
            self._expand_samples, 
            batched=True, 
            remove_columns=self.data_list.column_names,
            desc="Expanding samples..."
        )

    def _expand_samples(self, batch):
        result = {
            'id': [],
            'speaker': [],
            'language': [],
            'text': [],
            'entities': [],
            'reference_audio_path': [],
            'reference_audio_sample_rate': [],
            'audio_codes': [],
            'duration': [],
            "dataset_name": [],
        }
        
        batch_size = len(batch["id"])
        for i in range(batch_size):
            result['id'].append(f"{batch['id'][i]}_spoken")
            result['speaker'].append(batch['speaker'][i])
            result['language'].append(batch['language'][i])
            result['text'].append(batch['spoken_text'][i])
            result['entities'].append(None)
            result['reference_audio_path'].append(batch['reference_audio_path'][i])
            result['reference_audio_sample_rate'].append(batch['reference_audio_sample_rate'][i])
            result['audio_codes'].append(batch[f'audio_codes_{self.codec_name}'][i])
            result["duration"].append(batch['audio_duration'][i])
            result["dataset_name"].append(batch['dataset_name'][i])

            if batch['entities'][i] is not None and len(batch['entities'][i]) > 0:
                result['id'].append(f"{batch['id'][i]}_written")
                result['speaker'].append(batch['speaker'][i])
                result['language'].append(batch['language'][i])
                result['text'].append(batch['written_text'][i])
                result['entities'].append(batch['entities'][i])
                result['reference_audio_path'].append(batch['reference_audio_path'][i])
                result['reference_audio_sample_rate'].append(batch['reference_audio_sample_rate'][i])
                result['audio_codes'].append(batch[f'audio_codes_{self.codec_name}'][i])
                result["duration"].append(batch['audio_duration'][i])
                result["dataset_name"].append(batch['dataset_name'][i])
        
        return result
    
    def _prepare_entities_map_and_speaker_map(self):
        print("Building entity type vocabulary and speaker map from dataset...")
        entity_types = set()
        self.speaker_map = {}
        for sample in tqdm(self.data_list, desc="Scanning samples, preparing entities, and speakers..."):
            entities = sample.get("entities") or []
            for entity in entities:
                for prefix in ["B", "I", "E", "S"]:
                    entity_type = f"{prefix}-{entity['type']}"
                    entity_types.add(entity_type)
            speaker = sample.get("speaker", "")
            if speaker and speaker not in self.speaker_map:
                reference_audio_path = sample.get("reference_audio_path", "")
                reference_audio_sample_rate = sample.get("reference_audio_sample_rate", "")
                if reference_audio_path:
                    wav = self._load_audio_to_np(reference_audio_path, reference_audio_sample_rate)
                    wav = self._normalize(wav)
                    ref_mel = self.extract_mels(audio=wav)
                    self.speaker_map[speaker] = ref_mel

        for special_type in self.special_types:
            entity_types.add(special_type)
        
        entity_types = sorted(list(entity_types))
        self.entity_type_to_index = {entity_type: i for i, entity_type in enumerate(entity_types)}
        self.index_to_entity_type = {v: k for k, v in self.entity_type_to_index.items()}

        speaker_map = dict(sorted(self.speaker_map.items(), key=lambda x: x[0]))
        # Only remove columns that exist (dataset_root may not exist after expansion)
        columns_to_remove = [c for c in ["reference_audio_path", "reference_audio_sample_rate", "dataset_root"] 
                             if c in self.data_list.column_names]
        if columns_to_remove:
            self.data_list = self.data_list.remove_columns(columns_to_remove)


    def _prepare_text_and_entities(self, batch):
        all_text_ids = []
        all_entity_label_ids = []
        
        batch_size = len(batch["text"])
        for i in range(batch_size):
            text = batch["text"][i]
            entities = batch["entities"][i]
            text = self._build_assistant_text(text)
            text_ids, offsets = self._tokenize_texts(text)
            text_ids = text_ids[:, :-5]
            offsets = offsets[0, 3:-5] if offsets.dim() == 3 else offsets[3:-5]
            _, entity_label_ids = self._align_entities_with_text(offsets, entities)
            
            all_text_ids.append(text_ids.tolist())
            all_entity_label_ids.append(entity_label_ids)
        
        return {
            "text_ids": all_text_ids,
            "entity_label_ids": all_entity_label_ids,
        }


    def _align_entities_with_text(self, offset_mapping, entities):
        num_tokens = offset_mapping.shape[0]
        token_labels = ["PLAIN_WORD"] * num_tokens

        if entities is None or len(entities) == 0:
            return token_labels, [self.entity_type_to_index.get("PLAIN_WORD")] * num_tokens

        prefix_len = len("<|im_start|>assistant\n")
        token_spans = []
        for i, (ts, te) in enumerate(offset_mapping):
            if ts == te == 0:
                token_labels[i] = "SPECIAL"
                continue
            token_spans.append((i, ts, te))

        for entity in entities:
            ent_start = entity["start"] + prefix_len
            ent_end = entity["end"] + prefix_len
            ent_type = entity["type"]

            covered = []
            for idx, ts, te in token_spans:
                if ts < ent_end and te > ent_start:
                    covered.append(idx)

            if not covered:
                continue

            if len(covered) == 1:
                token_labels[covered[0]] = f"S-{ent_type}"
            else:
                token_labels[covered[0]] = f"B-{ent_type}"
                for mid in covered[1:-1]:
                    token_labels[mid] = f"I-{ent_type}"
                token_labels[covered[-1]] = f"E-{ent_type}"

        plain_idx = self.entity_type_to_index.get("PLAIN_WORD")
        token_label_ids = [self.entity_type_to_index.get(label, plain_idx) for label in token_labels]
        return token_labels, token_label_ids


    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str, sr: int) -> Tuple[np.ndarray, int]:
        audio, sr = librosa.load(x, sr=sr, mono=True)
        if sr != 24000:
            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=24000)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        return audio.astype(np.float32)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        max_val = np.abs(audio).max()
        if max_val > 1.0:
            audio = audio / max_val
        return audio
    
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
    def extract_mels(self, audio):
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

    def __getitem__(self, idx):
        item = self.data_list[idx]
        text_ids = item["text_ids"]
        audio_codes = item["audio_codes"]
        entity_label_ids = item["entity_label_ids"]

        speaker = item.get("speaker", "")
        ref_mel = self.speaker_map.get(speaker, None)

        text_ids = torch.tensor(text_ids, dtype=torch.long)
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        
        return {
            "id": item["id"],
            "text_ids": text_ids,    # 1 , t
            "audio_codes": audio_codes,      # t, 16
            "ref_mel": ref_mel,
            "entities": entity_label_ids
        }
       
    def collate_fn(self, batch):


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


class DistributedMixedSourceBatchSampler(Sampler):
    
    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_replicas = num_replicas if num_replicas is not None else 1
        self.rank = rank if rank is not None else 0
        self.shuffle = shuffle
        self.epoch = 0
        self.n_clean_samples_per_batch = max(1, int(self.batch_size * 0.3))
        remaining = self.batch_size - self.n_clean_samples_per_batch
        self.n_spoken_samples_per_batch = int(remaining * 0.4)
        self.n_written_samples_per_batch = remaining - self.n_spoken_samples_per_batch

        self.clean_indices = dataset.clean_samples.copy()
        self.spoken_indices = dataset.spoken_samples.copy()
        self.written_indices = dataset.written_samples.copy()
        
    def _build_batches(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        
        if self.shuffle:
            clean_perm = torch.randperm(len(self.clean_indices), generator=g).tolist()
            spoken_perm = torch.randperm(len(self.spoken_indices), generator=g).tolist()
            written_perm = torch.randperm(len(self.written_indices), generator=g).tolist()
            
            clean_pool = [self.clean_indices[i] for i in clean_perm]
            spoken_pool = [self.spoken_indices[i] for i in spoken_perm]
            written_pool = [self.written_indices[i] for i in written_perm]
        else:
            clean_pool = self.clean_indices.copy()
            spoken_pool = self.spoken_indices.copy()
            written_pool = self.written_indices.copy()
        
        batches = []
        clean_ptr, spoken_ptr, written_ptr = 0, 0, 0
        
        while True:
            batch = []
            clean_added = 0
            if clean_ptr >= len(clean_pool):
                clean_ptr = 0
            while clean_added < self.n_clean_samples_per_batch and clean_ptr < len(clean_pool):
                batch.append(clean_pool[clean_ptr])
                clean_ptr += 1
                clean_added += 1
            
            spoken_added = 0
            while spoken_added < self.n_spoken_samples_per_batch and spoken_ptr < len(spoken_pool):
                batch.append(spoken_pool[spoken_ptr])
                spoken_ptr += 1
                spoken_added += 1
            
            written_added = 0
            while written_added < self.n_written_samples_per_batch and written_ptr < len(written_pool):
                batch.append(written_pool[written_ptr])
                written_ptr += 1
                written_added += 1
            
            if len(batch) < self.batch_size:
                while len(batch) < self.batch_size:
                    if clean_ptr < len(clean_pool):
                        batch.append(clean_pool[clean_ptr])
                        clean_ptr += 1
                    elif spoken_ptr < len(spoken_pool):
                        batch.append(spoken_pool[spoken_ptr])
                        spoken_ptr += 1
                    elif written_ptr < len(written_pool):
                        batch.append(written_pool[written_ptr])
                        written_ptr += 1
                    else:
                        break
            
            if len(batch) < self.batch_size:
                break
            
            if self.shuffle:
                batch_perm = torch.randperm(len(batch), generator=g).tolist()
                batch = [batch[i] for i in batch_perm]
            
            batches.append(batch)
        
        if self.shuffle:
            batch_order = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_order]
        
        return batches
    
    def __iter__(self):
        batches = self._build_batches()
        min_batch_size = min(len(batch) for batch in batches)
        max_batch_size = max(len(batch) for batch in batches)
        print("Sampler Initialized.")
        print(f"Min batch size: {min_batch_size}")
        print(f"Max batch size: {max_batch_size}")
        rank_batches = batches[self.rank::self.num_replicas]
        
        for batch in rank_batches:
            yield batch
    
    def __len__(self):
        total_samples = len(self.clean_indices) + len(self.spoken_indices) + len(self.written_indices)
        total_batches = total_samples // self.batch_size
        return total_batches // self.num_replicas
    
    def set_epoch(self, epoch):
        self.epoch = epoch