import os
import argparse
import torch
import json
# from torch.utils.data import DataLoader
from train import improved_collate_fn, load_checkpoint
from models.graphllm import GraphLLM
from safetensors.torch import load_model
from train import evaluate, PlannerDataset, improved_collate_fn
import pickle
from torch.utils.data import Dataset, DataLoader

def evaluate_on_test_set(args, model, test_loader):
    """Runs evaluation on the test dataset and prints metrics"""
    metrics = evaluate(model, test_loader)
    print(f"Test Loss: {metrics['val_loss']:.4f}")
    
    # Save predictions for analysis
    output_path = os.path.join(args.output_dir, "test_predictions.json")
    with open(output_path, "w") as f:
        json.dump({"predictions": metrics["predictions"], "labels": metrics["labels"]}, f, indent=4)

    print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, default='/work/hdd/bcaq/ribhav/test_vanilla_WikiofGraph.pkl')
    parser.add_argument('--checkpoint_path', type=str, default='/work/hdd/bcaq/kagarwal2/best_model_aug.safetensors')
    parser.add_argument('--output_dir', type=str, default='/work/hdd/bcaq/kagarwal2/')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    # Load model
    print("Loading trained model...")
    model = GraphLLM(args)
    load_model(model, args.checkpoint_path)  # Load checkpoint
    model.eval()  # Set to evaluation mode

    # Load test data
    print("Loading test dataset...")
    with open(args.test_data_path, 'rb') as f:
        test_data = pickle.load(f)
    
    test_dataset = PlannerDataset(test_data)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=True, 
        collate_fn=improved_collate_fn
    )

    # Run evaluation
    print("Evaluating on test set...")
    evaluate_on_test_set(args, model, test_loader)

if __name__ == '__main__':
    main()
