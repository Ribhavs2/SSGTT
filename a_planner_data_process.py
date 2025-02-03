import json
import torch
from torch_geometric.data import Data
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import os
import random
import pickle
from math import factorial


from transformers import pipeline
from test_llama import load_pipeline, generate_paraphrases

# paraphraser = pipeline("text2text-generation", model="t5-base")

def generate_shuffled_graphs(batch_triples, m, batch_size):
    """
    Generate up to m unique shuffled variants for each list of triples in the batch.
    Args:
        batch_triples (list of lists): A batch of triples where each element is a list of triples.
        m (int): Number of unique shuffled variants per set of triples.
        batch_size (int): Number of items in the batch.
    Returns:
        list of lists: A batch of shuffled graphs corresponding to the input batch.
    """

    batch_shuffled_graphs = []

    for triples in batch_triples:
        # Remove duplicates to determine unique triples
        unique_triples = list(set(triples))

        # Calculate the maximum possible unique permutations
        max_possible_variants = factorial(len(unique_triples))

        # Ensure m does not exceed the number of unique permutations
        m_variants = min(m, max_possible_variants)

        unique_variants = set()
        while len(unique_variants) < m_variants:
            shuffled_triples = tuple(random.sample(unique_triples, len(unique_triples)))
            unique_variants.add(shuffled_triples)

        # Convert tuples back to lists for output
        batch_shuffled_graphs.append([list(shuffled) for shuffled in unique_variants])
    
    return batch_shuffled_graphs

def augment_graph_text_pair(batch_triples, batch_texts, m, n, pipeline):
    """
    Combine shuffled graphs and paraphrased texts into m*n augmented training pairs for a batch.
    Args:
        batch_triples (list of lists): A batch of triples where each element is a list of triples.
        batch_texts (list of str): Corresponding texts for each set of triples in the batch.
        m (int): Number of shuffled graph variants.
        n (int): Number of paraphrased text variants.
        pipeline: Paraphrasing model pipeline.
    Returns:
        list of dicts: Augmented pairs for the batch.
    """

    # Generate shuffled graphs for the batch
    batch_shuffled_graphs = generate_shuffled_graphs(batch_triples=batch_triples, m=m, batch_size=len(batch_triples))

    # Generate paraphrased texts for the batch
    batch_paraphrased_texts = generate_paraphrases(pipeline=pipeline, input_texts=batch_texts, batch_size = len(batch_triples), num_repeats = n)

    augmented_batch = []

    for i in range(len(batch_triples)):
        shuffled_graphs = batch_shuffled_graphs[i]
        paraphrased_texts = batch_paraphrased_texts[i]
        
        for shuffled_graph in shuffled_graphs:
            for paraphrased_text in paraphrased_texts:
                augmented_batch.append({"triplet": shuffled_graph, "text": paraphrased_text})
            # augmented_batch.append({"triplet": shuffled_graph, "text": batch_texts[i]})
    
    return augmented_batch


class GraphProcessor:
    def __init__(self, bert_model='sentence-transformers/all-roberta-large-v1'):
        """
        Initialize the data processor.
        Args:
            bert_model (str): Name of the SentenceBERT model to use for embeddings
        """
        self.sentence_model = SentenceTransformer(bert_model)
        self.embed_dim = self.sentence_model.get_sentence_embedding_dimension()
    

    def create_graph_from_triples(self, triple_strs):
        """Convert a list of triple strings into a PyG graph with predicate encodings"""
        nodes = set()
        edge_triples = []
        
        # Collect unique nodes and edges
        for triple_str in triple_strs:
            # Keep original triple string for description
            triple_str = triple_str.strip('()')
            parts = triple_str.split('|')
            
            # Extract subject, predicate, object
            subject = parts[0].replace('S>', '').strip()
            predicate = parts[1].replace('P>', '').strip()
            object_ = parts[2].replace('O>', '').strip()
            
            nodes.add(subject)
            nodes.add(object_)
            edge_triples.append((subject, predicate, object_))
        
        # Create node mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Create edge index and collect predicates
        edge_index = []
        predicates = []
        
        for subj, pred, obj in edge_triples:
            # Add forward edge
            edge_index.append([node_to_idx[subj], node_to_idx[obj]])
            predicates.append(pred)  # Original predicate
            
            # Add reverse edge
            edge_index.append([node_to_idx[obj], node_to_idx[subj]])
            predicates.append(f"inverse_{pred}")  # Inverse predicate
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Create node features (embeddings)
        node_texts = list(nodes)
        node_embeddings = self.sentence_model.encode(node_texts)
        node_features = torch.tensor(node_embeddings, dtype=torch.float)
        
        # Create edge features (only encode predicates)
        predicate_embeddings = self.sentence_model.encode(predicates)
        edge_features = torch.tensor(predicate_embeddings, dtype=torch.float)
        
        # Create graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features
        )
        
        return graph, triple_strs  # Return original triple strings for description

def process_batch(batch):
    """
    Process a batch to generate graphs without augmentation.
    """
    batch_triples = [item['triplet'].strip('()').split('), ') for item in batch]
    batch_triples = [[triple.strip('()') for triple in triples] for triples in batch_triples]
    batch_texts = [item['text'] for item in batch]

    processed_batch = []
    graph_processor = GraphProcessor()

    for triples, text in zip(batch_triples, batch_texts):
        try:
            graph, original_triples = graph_processor.create_graph_from_triples(triples)
            processed_batch.append({'input': original_triples, 'label': text, 'graphs': [graph]})
        except Exception as e:
            print(f"Error generating graph: {e}")
            continue

    return processed_batch


def main():
    # Paths and parameters
    dataset_path = r'Data/WikiOFGraph_train.jsonl'
    output_dir = r'Processed_Data'
    m = 2  # Number of shuffled graph variants
    n = 2  # Number of paraphrased text variants
    # batch_size = 5000  # Batch size for processing
    # vanilla_size = 1_000_000  # Number of samples for the vanilla dataset
    # augmented_size = 80_000  # Number of samples for the augmented dataset

    batch_size = 5  # Batch size for processing
    vanilla_size = 20  # Number of samples for the vanilla dataset
    augmented_size = 10  # Number of samples for the augmented dataset

    # Load the dataset
    print("Loading original dataset ...")
    with open(dataset_path, 'r') as f:
        wikiofgraph_data = [json.loads(line) for line in f]

    # Sample 1 million random samples for the vanilla dataset
    print("Selecting 1 million samples for the vanilla dataset ...")
    vanilla_samples = random.sample(wikiofgraph_data, vanilla_size)

    # Process and store the vanilla dataset
    print("Generating graphs for the vanilla dataset ...")
    vanilla_data = []
    for i in tqdm(range(0, len(vanilla_samples), batch_size)):
        batch = vanilla_samples[i:i + batch_size]
        vanilla_data.extend(process_batch(batch))

    print(f"Vanilla dataset size: {len(vanilla_data)}")

    # Save the vanilla dataset
    os.makedirs(output_dir, exist_ok=True)
    vanilla_jsonl_path = os.path.join(output_dir, 'vanilla_WikiofGraph.jsonl')
    vanilla_pickle_path = os.path.join(output_dir, 'vanilla_WikiofGraph.pkl')

    with open(vanilla_jsonl_path, 'w') as f:
        for pair in vanilla_data:
            f.write(json.dumps(pair, default=str) + '\n')

    with open(vanilla_pickle_path, 'wb') as f:
        pickle.dump(vanilla_data, f)

    print(f"Vanilla dataset saved to {vanilla_jsonl_path} and {vanilla_pickle_path}")

    # Sample 80,000 for augmentation from the vanilla dataset
    print("Selecting 80,000 samples for augmentation ...")
    augmentation_samples = random.sample(vanilla_data, augmented_size)

    model_id = "./models/Llama-3.1-8B-Instruct"
    pipeline = load_pipeline(model_id)

    # Process and augment the selected samples
    augmented_data = []
    print("Augmenting graph-text pairs ...")
    graph_processor = GraphProcessor()  # Reuse the same processor

    for i in tqdm(range(0, len(augmentation_samples), batch_size)):
        batch = augmentation_samples[i:i + batch_size]
        batch_triples = [item['input'] for item in batch]
        batch_texts = [item['label'] for item in batch]
        augmented_batch = augment_graph_text_pair(batch_triples, batch_texts, m, n, pipeline)

        # Generate graphs for augmented pairs
        for augmented_item in augmented_batch:
            try:
                graph, original_triples = graph_processor.create_graph_from_triples(augmented_item["triplet"])
                augmented_item["graphs"] = [graph]
            except Exception as e:
                print(f"Error generating graph for augmented item: {e}")
                continue

        augmented_data.extend(augmented_batch)

    print(f"Augmented dataset size: {len(augmented_data)}")

    # Save the augmented dataset
    augmented_jsonl_path = os.path.join(output_dir, 'augmented_WikiofGraph.jsonl')
    augmented_pickle_path = os.path.join(output_dir, 'augmented_WikiofGraph.pkl')

    with open(augmented_jsonl_path, 'w') as f:
        for pair in augmented_data:
            f.write(json.dumps(pair, default=str) + '\n')

    with open(augmented_pickle_path, 'wb') as f:
        pickle.dump(augmented_data, f)

    print(f"Augmented dataset saved to {augmented_jsonl_path} and {augmented_pickle_path}")



if __name__ == "__main__":
    main()
