#!/usr/bin/env python3
# coding=utf-8
"""
Convert JSONL TTS metadata to HuggingFace Dataset format.

This script:
1. Reads a JSONL file with TTS sample metadata
2. Converts field names (written -> written_text, spoken -> spoken_text)
3. Copies audio files to dataset folder with relative paths
4. Stores audio codes inline (as 2D integer sequences)
5. Creates per-speaker reference audio mappings
6. Outputs a sharded HuggingFace Dataset ready for Hub upload

Usage:
    python convert_to_hf_dataset.py \
        --input_jsonl metadata.jsonl \
        --output_dir ./hf_dataset \
        --codec_name qwen3_12hz \
        --num_shards 100

To push to HuggingFace Hub:
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_folder(
        folder_path="./hf_dataset",
        repo_id="your-username/dataset-name",
        repo_type="dataset"
    )
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List
import soundfile as sf
from tqdm import tqdm
from datasets import Dataset, Features, Value, Sequence
import shutil


def get_audio_info(audio_path: str) -> Dict[str, Any]:
    """Extract duration and sample rate from an audio file."""
    try:
        info = sf.info(audio_path)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate
        }
    except Exception as e:
        print(f"Warning: Could not read audio info from {audio_path}: {e}")
        return {"duration": 0.0, "sample_rate": 0}


def get_speaker_reference_audios(
    jsonl_path: str,
    reference_audio_dir: Path
) -> Dict[str, Dict[str, Any]]:
    """
    Get one reference audio per speaker (duration > 3s).
    Copies the audio file to reference_audio_dir.
    Returns a dict mapping speaker name to their reference info.
    """
    speaker_refs = {}
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            speaker = sample.get('speaker', '')
            duration = sample.get('duration', 0.0)
            
            if speaker and speaker not in speaker_refs and duration > 3.0:
                audio_path = sample.get('audio', sample.get('audio_path', ''))
                if audio_path:
                    # Copy the audio to the reference audio directory
                    dest_filename = f"{speaker}.wav"
                    dest_path = reference_audio_dir / dest_filename
                    shutil.copy(audio_path, dest_path)
                    
                    # Get audio info
                    audio_info = get_audio_info(str(dest_path))
                    
                    speaker_refs[speaker] = {
                        "relative_path": f"reference_audio/{dest_filename}",
                        "duration": audio_info["duration"],
                        "sample_rate": audio_info["sample_rate"]
                    }
    
    return speaker_refs


def convert_sample(
    sample: Dict[str, Any],
    codec_name: str,
    speaker_refs: Dict[str, Dict[str, Any]],
    audio_dir: Path
) -> Dict[str, Any]:
    """Convert a single sample to the new format."""
    
    sample_id = sample.get('id', '')
    speaker = sample.get('speaker', '')
    
    # Get audio path and copy to dataset folder
    audio_path = sample.get('audio')
    audio_relative_path = ""
    audio_info = {"duration": 0.0, "sample_rate": 0}
    
    if audio_path:
        assert os.path.isabs(audio_path), "Audio path must be absolute"
        # Use id_speaker to avoid filename collisions (same id for different speakers)
        dest_filename = f"{sample_id}_{speaker}.wav"
        dest_path = audio_dir / dest_filename
        shutil.copy(audio_path, dest_path)
        audio_relative_path = f"audio/{dest_filename}"
        audio_info = get_audio_info(str(dest_path))
    
    # Get reference audio info for this speaker
    ref_info = speaker_refs.get(speaker, {
        "relative_path": "",
        "duration": 0.0,
        "sample_rate": 0
    })
    
    # Get audio codes (store inline as 2D list)
    audio_codes = sample.get('audio_codes', [])
    
    # Get entities (list of dicts with type, start, end)
    # If not present, use empty list
    entities = sample.get('entities', [])
    
    # Build the new sample structure
    new_sample = {
        "id": sample_id,
        "speaker": speaker,
        "language": sample.get('language', 'English'),
        "written_text": sample.get('written', ''),
        "spoken_text": sample.get('spoken', ''),
        "entities": entities,  # List of {type, start, end} dicts
        "audio_path": audio_relative_path,
        "audio_duration": audio_info["duration"],
        "audio_sample_rate": audio_info["sample_rate"],
        "reference_audio_path": ref_info["relative_path"],
        "reference_audio_duration": ref_info["duration"],
        "reference_audio_sample_rate": ref_info["sample_rate"],
        f"audio_codes_{codec_name}": audio_codes,  # Stored inline as 2D list
    }
    
    return new_sample


def create_dataset_features(codec_name: str) -> Features:
    """Create HuggingFace Dataset features schema."""
    return Features({
        'id': Value('string'),
        'speaker': Value('string'),
        'language': Value('string'),
        'written_text': Value('string'),
        'spoken_text': Value('string'),
        # Entities: list of {type: str, start: int, end: int}
        # Use list syntax [{...}] for sequence of dicts (not Sequence({...}))
        'entities': [{
            'type': Value('string'),
            'start': Value('int32'),
            'end': Value('int32'),
        }],
        'audio_path': Value('string'),  # Relative path: audio/id_speaker.wav
        'audio_duration': Value('float32'),
        'audio_sample_rate': Value('int32'),
        'reference_audio_path': Value('string'),  # Relative path: reference_audio/speaker.wav
        'reference_audio_duration': Value('float32'),
        'reference_audio_sample_rate': Value('int32'),
        # Audio codes stored inline as 2D sequence: [num_layers, num_tokens]
        f'audio_codes_{codec_name}': Sequence(Sequence(Value('int16'))),
    })


def process_jsonl(
    input_jsonl: str,
    output_dir: str,
    codec_name: str,
    num_shards: int = 100
):
    """Main processing function."""
    
    input_path = Path(input_jsonl)
    output_path = Path(output_dir)
    
    # Reference audio paths
    reference_audio_dir = output_path / "reference_audio"
    
    # Audio dir for all sample audio files
    audio_dir = output_path / "audio"
    
    # Create output directories
    output_path.mkdir(parents=True, exist_ok=True)
    reference_audio_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Finding reference audio for each speaker (duration > 3s)...")
    speaker_refs = get_speaker_reference_audios(input_jsonl, reference_audio_dir)
    print(f"  Found {len(speaker_refs)} speakers: {list(speaker_refs.keys())}")
    for speaker, info in speaker_refs.items():
        print(f"    {speaker}: {info['relative_path']} ({info['duration']:.2f}s)")
    
    # Count total lines for progress bar
    print("\nStep 2: Counting samples...")
    with open(input_jsonl, 'r') as f:
        total_lines = sum(1 for line in f if line.strip())
    print(f"  Total samples: {total_lines}")
    
    # Process samples
    print("\nStep 3: Converting samples (copying audio + extracting codes)...")
    
    def sample_generator():
        with open(input_jsonl, 'r') as f:
            for line in tqdm(f, total=total_lines, desc="Processing"):
                if not line.strip():
                    continue
                sample = json.loads(line)
                yield convert_sample(
                    sample,
                    codec_name=codec_name,
                    speaker_refs=speaker_refs,
                    audio_dir=audio_dir
                )
    
    # Create dataset from generator
    print("\nStep 4: Creating HuggingFace Dataset...")
    features = create_dataset_features(codec_name)
    dataset = Dataset.from_generator(
        sample_generator,
        features=features
    )
    
    # Save dataset
    print(f"\nStep 5: Saving dataset to {output_path} with {num_shards} shards...")
    dataset.save_to_disk(str(output_path), num_shards=num_shards)
    
    # Save metadata about the dataset
    metadata = {
        "total_samples": len(dataset),
        "num_shards": num_shards,
        "codecs": [codec_name],
        "speakers": list(speaker_refs.keys()),
        "speaker_references": speaker_refs,
        "paths_info": {
            "audio_dir": "audio/",
            "reference_audio_dir": "reference_audio/",
            "note": "All paths in the dataset are relative to the dataset root directory"
        }
    }
    
    with open(output_path / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Done! Dataset saved to {output_path}")
    print(f"{'='*60}")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Speakers: {list(speaker_refs.keys())}")
    print(f"  Codec: {codec_name}")
    print(f"\nDataset structure:")
    print(f"  {output_path}/")
    print(f"  ├── data-00000-of-{num_shards:05d}.arrow")
    print(f"  ├── ...")
    print(f"  ├── dataset_info.json")
    print(f"  ├── dataset_metadata.json")
    print(f"  ├── audio/")
    print(f"  │   ├── tn_0000001_alba.wav")
    print(f"  │   └── ...")
    print(f"  └── reference_audio/")
    print(f"      ├── alba.wav")
    print(f"      └── ...")
    print(f"\nTo push to HuggingFace Hub:")
    print(f"  from huggingface_hub import HfApi")
    print(f"  api = HfApi()")
    print(f"  api.upload_folder(")
    print(f"      folder_path='{output_path}',")
    print(f"      repo_id='your-username/dataset-name',")
    print(f"      repo_type='dataset'")
    print(f"  )")
    
    return dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert JSONL TTS metadata to HuggingFace Dataset format"
    )
    
    parser.add_argument(
        '--input_jsonl', '-i',
        type=str,
        required=True,
        help='Path to input JSONL file'
    )
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        required=True,
        help='Output directory for HuggingFace dataset'
    )
    parser.add_argument(
        '--codec_name',
        type=str,
        default='qwen3_12hz',
        help='Name of the codec (default: qwen3_12hz)'
    )
    parser.add_argument(
        '--num_shards',
        type=int,
        default=100,
        help='Number of shards for the dataset (default: 100)'
    )
    
    args = parser.parse_args()
    
    process_jsonl(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        codec_name=args.codec_name,
        num_shards=args.num_shards
    )


if __name__ == "__main__":
    main()
