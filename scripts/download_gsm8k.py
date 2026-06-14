#!/usr/bin/env python3
"""
GSM8K Dataset Preparation for RIN Fidelity Testing

Descarga y prepara el dataset GSM8K para validar fidelidad del modelo.
GSM8K contiene 8,500 problemas de matemáticas de nivel escolar.
"""

import json
import sys

try:
    from datasets import load_dataset
except ImportError:
    print("ERROR: datasets library not installed")
    print("Run: pip3 install datasets")
    sys.exit(1)

def download_gsm8k():
    print("Downloading GSM8K dataset...")
    print("This may take a few minutes...")
    
    try:
        # Download GSM8K from HuggingFace
        dataset = load_dataset("gsm8k", "main")
        
        print(f"\nDataset loaded successfully!")
        print(f"Train examples: {len(dataset['train'])}")
        print(f"Test examples: {len(dataset['test'])}")
        
        # Save a subset for quick testing
        print("\nSaving subset for RIN testing...")
        
        # Save first 100 examples as test set
        test_subset = []
        for i, example in enumerate(dataset['test']):
            if i >= 100:
                break
            test_subset.append({
                'id': i,
                'question': example['question'],
                'answer': example['answer']
            })
        
        with open('gsm8k_test_100.json', 'w') as f:
            json.dump(test_subset, f, indent=2)
        
        print(f"Saved 100 test examples to gsm8k_test_100.json")
        
        # Show sample
        print("\nSample question:")
        print(test_subset[0]['question'][:200] + "...")
        print("\nSample answer:")
        print(test_subset[0]['answer'][:100] + "...")
        
        return True
        
    except Exception as e:
        print(f"ERROR downloading dataset: {e}")
        print("\nFalling back to creating synthetic test data...")
        
        # Create synthetic math problems if download fails
        synthetic_data = []
        for i in range(100):
            synthetic_data.append({
                'id': i,
                'question': f"What is {i} + {i*2}?",
                'answer': str(i + i*2)
            })
        
        with open('gsm8k_test_100.json', 'w') as f:
            json.dump(synthetic_data, f, indent=2)
        
        print("Created synthetic test data (100 math problems)")
        return False

if __name__ == "__main__":
    success = download_gsm8k()
    sys.exit(0 if success else 1)
