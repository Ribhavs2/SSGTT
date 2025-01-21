import torch
from argparse import Namespace
from graphllm_ans_v2 import GraphLLM  
from torch_geometric.data import Data 


def create_synthetic_graph_data():
    """
    Create synthetic graph data for testing purposes.
    """
    # Node features (x), edge index, and edge attributes
    x = torch.randn((5, 3))  # 5 nodes, each with 3 features
    edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]])  # Directed edges
    edge_attr = torch.randn((5, 2))  # 5 edges, each with 2 features

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def main():
    # Define the arguments needed for the GraphLLM class
    args = Namespace(
        llm_model_path="gpt2",  # Pre-trained model path (can be any LM)
        max_txt_len=128, 
        max_new_tokens=50,
        llm_frozen="True",
        finetune_method="lora",  # Fine-tuning method 
        lora_r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        gnn_model_name="gat",  # Graph Neural Network model, tweakable
        gnn_in_dim=3,  # Input dimension for GNN (matches node features in graph data)
        gnn_hidden_dim=128,
        gnn_num_layers=2, 
        gnn_dropout=0.1, 
        gnn_num_heads=4, 
    )


    # Initialize the GraphLLM model
    model = GraphLLM(args)

    # Print model details (optional)
    model.print_trainable_params()

    # Create synthetic graph data
    graph_data = create_synthetic_graph_data() # we can replace this with the actual graph data
    graphs = [graph_data]  # Single graph for this example

    # Create synthetic input samples
    samples = {
        "input": ["Question: What is the capital of France?"],
        "label": ["[SUFFICIENT] The capital of France is Paris."],
        "graphs": [graphs],
    }

    # Test the forward method (training)
    model.train()  # Set the model to training mode
    loss = model.forward(samples)
    print(f"Training Loss: {loss.item()}")

    # Test the inference method (generation)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        output = model.inference(samples)
    print(f"Inference Output: {output}")

if __name__ == "__main__":
    main()
