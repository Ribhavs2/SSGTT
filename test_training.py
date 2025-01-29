import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_scheduler

# Dataset Class
class GraphTextDataset(Dataset):
    def __init__(self, json_path, tokenizer, max_txt_len, max_new_tokens):
        with open(json_path, 'r') as f:
            self.data = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input": item["input"],  # Graph triples
            "label": item["label"],  # Ground truth text
            "graphs": item["graphs"],  # Preprocessed graphs (PyG Data objects)
        }


# Training Function
def train_model(
    model,
    dataset,
    batch_size,
    lr,
    epochs,
    device,
    gradient_accumulation_steps=1,
    log_interval=10
):
    # DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=None)

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(dataloader) // gradient_accumulation_steps
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Move model to device
    model.to(device)
    model.train()

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            # Move batch data to the device
            batch = {key: value for key, value in batch.items()}
            batch["graphs"] = [g.to(device) for g in batch["graphs"]]

            # Forward pass
            loss = model.forward(batch)
            loss = loss / gradient_accumulation_steps
            loss.backward()
            total_loss += loss.item()

            # Gradient accumulation and optimizer step
            if (step + 1) % gradient_accumulation_steps == 0 or step == len(dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            if step % log_interval == 0:
                print(f"Epoch: {epoch+1}, Step: {step}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # End of epoch
        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(dataloader):.4f}")


# Main Script to Train the Model
if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "Processed_Data/processed_WikiofGraph_with_graphs.jsonl"
    MODEL_PATH = "./models/Llama-3.1-8B-Instruct"
    OUTPUT_DIR = "./checkpoints"
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-5
    EPOCHS = 5
    MAX_TXT_LEN = 128
    MAX_NEW_TOKENS = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize tokenizer and dataset
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)

    dataset = GraphTextDataset(
        json_path=DATASET_PATH,
        tokenizer=tokenizer,
        max_txt_len=MAX_TXT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    # Initialize model
    from graphllm_ans_v2 import GraphLLM
    from argparse import Namespace
    args = Namespace(
        max_txt_len=MAX_TXT_LEN,
        max_new_tokens=MAX_NEW_TOKENS,
        llm_model_path=MODEL_PATH,
        llm_frozen="True",  # Change to "False" if you want to finetune the LLM
        finetune_method="lora",  # Choose "full" or "lora"
        gnn_model_name="gt",  # Change based on GNN model
        gnn_in_dim=1024,  # Match with dataset embeddings
        gnn_hidden_dim=1024,
        gnn_num_layers=2,
        gnn_dropout=0.1,
        gnn_num_heads=4,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = GraphLLM(args)

    # Train the model
    train_model(
        model=model,
        dataset=dataset,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        epochs=EPOCHS,
        device=DEVICE,
        gradient_accumulation_steps=2,  # Adjust for your resources
        log_interval=5,  # Log every 5 steps
    )

    # Save the model
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/graphllm_checkpoint.pt")
    print(f"Model saved to {OUTPUT_DIR}/graphllm_checkpoint.pt")
