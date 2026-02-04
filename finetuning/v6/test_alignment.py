#!/usr/bin/env python3
"""
Test script to verify text token and entity alignment in TTSDataset.
Creates a mock 4-sample dataset and checks if entities align correctly with tokens.
"""
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

def create_mock_dataset(tmp_dir):
    """Create a small mock dataset with 4 samples containing known entities."""
    
    # Create a dummy audio file (24kHz, 3 seconds of silence)
    dummy_audio_path = os.path.join(tmp_dir, "dummy_audio.wav")
    import soundfile as sf
    dummy_audio = np.zeros(24000 * 3, dtype=np.float32)  # 3 seconds at 24kHz
    sf.write(dummy_audio_path, dummy_audio, 24000)
    
    # Define test samples with known entity positions
    samples = [
        {
            "id": "sample_001",
            "speaker": "speaker_A",
            "language": "en",
            "spoken_text": "Call me at one two three four five six seven eight nine zero.",
            "written_text": "Call me at 1234567890.",
            "entities": [
                {"start": 11, "end": 21, "type": "PHONE_NUMBER"}  # "1234567890"
            ],
            "reference_audio_path": dummy_audio_path,
            "reference_audio_sample_rate": 24000,
            "audio_duration": 3.0,
            "audio_codes_qwen3_12hz": np.random.randint(0, 1000, (36, 16)).tolist(),
        },
        {
            "id": "sample_002",
            "speaker": "speaker_A",
            "language": "en",
            "spoken_text": "The date is January first twenty twenty five.",
            "written_text": "The date is 01/01/2025.",
            "entities": [
                {"start": 12, "end": 22, "type": "DATE"}  # "01/01/2025"
            ],
            "reference_audio_path": dummy_audio_path,
            "reference_audio_sample_rate": 24000,
            "audio_duration": 3.5,
            "audio_codes_qwen3_12hz": np.random.randint(0, 1000, (42, 16)).tolist(),
        },
        {
            "id": "sample_003",
            "speaker": "speaker_B",
            "language": "en",
            "spoken_text": "I have one hundred dollars and fifty cents.",
            "written_text": "I have $100.50.",
            "entities": [
                {"start": 7, "end": 14, "type": "MONEY"}  # "$100.50"
            ],
            "reference_audio_path": dummy_audio_path,
            "reference_audio_sample_rate": 24000,
            "audio_duration": 4.0,
            "audio_codes_qwen3_12hz": np.random.randint(0, 1000, (48, 16)).tolist(),
        },
        {
            "id": "sample_004",
            "speaker": "speaker_B",
            "language": "en",
            "spoken_text": "Send to example at email dot com and one two three.",
            "written_text": "Send to example@email.com and 123.",
            "entities": [
                {"start": 8, "end": 25, "type": "EMAIL"},     # "example@email.com"
                {"start": 30, "end": 33, "type": "CARDINAL"}  # "123"
            ],
            "reference_audio_path": dummy_audio_path,
            "reference_audio_sample_rate": 24000,
            "audio_duration": 5.0,
            "audio_codes_qwen3_12hz": np.random.randint(0, 1000, (60, 16)).tolist(),
        },
    ]
    
    # Create HuggingFace dataset
    hf_dataset = Dataset.from_list(samples)
    
    # Save to disk
    dataset_path = os.path.join(tmp_dir, "test_dataset")
    hf_dataset.save_to_disk(dataset_path)
    
    return dataset_path, samples


def test_alignment_standalone(processor):
    """Test alignment logic in isolation without full dataset loading."""
    
    print("\n" + "="*80)
    print("STANDALONE ALIGNMENT TEST")
    print("="*80)
    
    # Test cases with known entity positions
    test_cases = [
        {
            "text": "Call me at 1234567890.",
            "entities": [{"start": 11, "end": 21, "type": "PHONE_NUMBER"}],
        },
        {
            "text": "Send to example@email.com and 123.",
            "entities": [
                {"start": 8, "end": 25, "type": "EMAIL"},
                {"start": 30, "end": 33, "type": "CARDINAL"}
            ],
        },
        {
            "text": "I have $100.50.",
            "entities": [{"start": 7, "end": 14, "type": "MONEY"}],
        },
    ]
    
    # Build entity type to index map
    entity_types = set()
    for tc in test_cases:
        for ent in tc["entities"]:
            for prefix in ["B", "I", "E", "S"]:
                entity_types.add(f"{prefix}-{ent['type']}")
    
    special_types = ["SPECIAL", "PLAIN_WORD", "PLAIN"]
    for st in special_types:
        entity_types.add(st)
    
    entity_types = sorted(list(entity_types))
    entity_type_to_index = {et: i for i, et in enumerate(entity_types)}
    index_to_entity_type = {v: k for k, v in entity_type_to_index.items()}
    
    print(f"\nEntity vocabulary: {entity_type_to_index}")
    
    for i, tc in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i+1}")
        print(f"{'='*60}")
        
        text = tc["text"]
        entities = tc["entities"]
        
        # Build assistant text (same as dataset.py)
        assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        
        print(f"Original text: '{text}'")
        print(f"Entities: {entities}")
        
        # Tokenize
        tokenized = processor(
            text=assistant_text, 
            return_tensors="pt", 
            padding=True, 
            return_offsets_mapping=True
        )
        
        input_ids = tokenized["input_ids"]
        offsets = tokenized["offset_mapping"]
        
        # Apply same slicing as dataset.py
        # text_ids keeps first 3 tokens but offsets skips them
        # In collate_fn, text_ids[0, 3:] is used which aligns with offsets[3:-5]
        full_input_ids = input_ids.clone()
        input_ids_sliced = input_ids[:, :-5]  # Remove last 5 tokens
        offsets = offsets[0, 3:-5] if offsets.dim() == 3 else offsets[3:-5]  # Skip first 3, remove last 5
        
        num_tokens = offsets.shape[0]
        token_labels = ["PLAIN_WORD"] * num_tokens
        
        # Decode tokens that MATCH the offsets (tokens 3 to N-6, same as text_ids[0, 3:] used in collate)
        aligned_token_ids = input_ids_sliced[0, 3:].tolist()  # Skip first 3 to match offsets
        tokens = processor.tokenizer.convert_ids_to_tokens(aligned_token_ids)
        
        print(f"\nTokenization (after slicing):")
        print(f"  Number of tokens: {num_tokens}")
        print(f"  Number of entity labels: {len(token_labels)}")
        
        # Alignment logic (same as dataset.py)
        prefix_len = len("<|im_start|>assistant\n")
        token_spans = []
        
        for idx, (ts, te) in enumerate(offsets.tolist()):
            if ts == te == 0:
                token_labels[idx] = "SPECIAL"
                continue
            token_spans.append((idx, ts, te))
        
        for entity in entities:
            ent_start = entity["start"] + prefix_len
            ent_end = entity["end"] + prefix_len
            ent_type = entity["type"]
            
            covered = []
            for idx, ts, te in token_spans:
                if ts < ent_end and te > ent_start:
                    covered.append(idx)
            
            if not covered:
                print(f"\n  WARNING: Entity '{ent_type}' at [{entity['start']}, {entity['end']}] has no token coverage!")
                continue
            
            if len(covered) == 1:
                token_labels[covered[0]] = f"S-{ent_type}"
            else:
                token_labels[covered[0]] = f"B-{ent_type}"
                for mid in covered[1:-1]:
                    token_labels[mid] = f"I-{ent_type}"
                token_labels[covered[-1]] = f"E-{ent_type}"
        
        # Convert to indices
        plain_idx = entity_type_to_index.get("PLAIN_WORD")
        token_label_ids = [entity_type_to_index.get(label, plain_idx) for label in token_labels]
        
        # Print alignment results
        print(f"\nToken-Entity Alignment:")
        print("-" * 80)
        print(f"{'Idx':>4} | {'Token':<20} | {'Offset':>12} | {'Label':<20} | {'ID':>4}")
        print("-" * 80)
        
        for idx, (token, (ts, te), label, label_id) in enumerate(zip(tokens, offsets.tolist(), token_labels, token_label_ids)):
            # Highlight entity tokens
            marker = "***" if label not in ["PLAIN_WORD", "SPECIAL"] else ""
            print(f"{idx:>4} | {repr(token):<20} | ({ts:>4}, {te:>4}) | {label:<20} | {label_id:>4} {marker}")
        
        # Verify entity coverage
        print(f"\n✓ Verification:")
        for entity in entities:
            ent_text = text[entity["start"]:entity["end"]]
            ent_labels = [l for l in token_labels if entity["type"] in l]
            print(f"  Entity '{ent_text}' ({entity['type']}): {len(ent_labels)} tokens tagged")
            if ent_labels:
                print(f"    Tags: {ent_labels}")
    
    return True


def test_full_dataset(processor, config, tmp_dir):
    """Test the full TTSDataset with mock data."""
    
    print("\n" + "="*80)
    print("FULL DATASET TEST")
    print("="*80)
    
    # Import here to avoid circular imports
    from dataset import TTSDataset
    
    # Create mock dataset
    dataset_path, original_samples = create_mock_dataset(tmp_dir)
    
    print(f"\nCreated mock dataset at: {dataset_path}")
    print(f"Number of original samples: {len(original_samples)}")
    
    # Load dataset
    dataset_paths = {"test": dataset_path}
    
    try:
        dataset = TTSDataset(
            dataset_paths=dataset_paths,
            processor=processor,
            config=config,
            codec_name="qwen3_12hz"
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"Total expanded samples: {len(dataset)}")  # Should be 8 (4 * 2)
        print(f"Entity types: {list(dataset.entity_type_to_index.keys())}")
        
        # Check a few samples
        print("\n" + "-"*60)
        print("SAMPLE INSPECTION")
        print("-"*60)
        
        for i in range(min(4, len(dataset))):
            item = dataset[i]
            sample_id = dataset.data_list[i]["id"]
            
            print(f"\n--- Sample {i}: {sample_id} ---")
            print(f"  text_ids shape: {item['text_ids'].shape}")
            print(f"  audio_codes shape: {item['audio_codes'].shape}")
            print(f"  entities length: {len(item['entities'])}")
            
            # Decode text to verify - entity labels align with text_ids[0, 3:] (skip first 3 tokens)
            all_text_tokens = processor.tokenizer.convert_ids_to_tokens(item['text_ids'][0].tolist())
            aligned_tokens = all_text_tokens[3:]  # Skip first 3 to match entity_label_ids
            
            # Show alignment for first few tokens
            print(f"\n  Token-Entity alignment (text_ids[3:] aligned with entity_labels):")
            for j, (token, ent_id) in enumerate(zip(aligned_tokens[:15], item['entities'][:15])):
                ent_label = dataset.index_to_entity_type.get(ent_id, "UNKNOWN")
                marker = "***" if ent_label not in ["PLAIN_WORD", "SPECIAL"] else ""
                print(f"    {j:>3}: {repr(token):<15} -> {ent_label:<20} {marker}")
        
        # Test collate function
        print("\n" + "-"*60)
        print("COLLATE FUNCTION TEST")
        print("-"*60)
        
        batch_items = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = dataset.collate_fn(batch_items)
        
        print(f"\nBatch shapes:")
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: {v.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*80)
    print("TTS DATASET ALIGNMENT TEST")
    print("="*80)
    
    MODEL_PATH = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    
    print(f"\nLoading model from: {MODEL_PATH}")
    
    # Load model and processor
    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
    )
    
    processor = qwen3tts.processor
    config = qwen3tts.model.config
    
    print(f"Processor loaded successfully")
    print(f"Tokenizer is fast: {processor.tokenizer.is_fast}")
    
    # Test 1: Standalone alignment test
    test_alignment_standalone(processor)
    
    # Test 2: Full dataset test
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_full_dataset(processor, config, tmp_dir)
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
