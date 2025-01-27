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


def process_batch(batch, m, n, pipeline):
    """
    Process a batch of items by performing augmentation for the whole batch.
    """
    batch_triples = [item['triplet'].strip('()').split('), ') for item in batch]
    batch_triples = [[triple.strip('()') for triple in triples] for triples in batch_triples]
    batch_texts = [item['text'] for item in batch]

    # Generate augmented pairs for the batch
    try:
        augmented_batch = augment_graph_text_pair(batch_triples, batch_texts, m, n, pipeline)
    except Exception as e:
        print(f"Error augmenting batch: {e}")
        augmented_batch = []
    
    return augmented_batch


def main():
    # Define paths and augmentation parameters
    # dataset_path = r'Data\WikiOFGraph-test.jsonl'
    dataset_path = r'Data/WikiOFGraph-test.jsonl'
    output_dir = r'Processed_Data'
    m = 3  # Number of shuffled graph variants
    n = 3  # Number of paraphrased text variants
    batch_size = 5  # Batch size for augmentation

    # Load the WikiofGraph dataset
    print("Loading WikiofGraph data ...")
    with open(dataset_path, 'r') as f:
        wikiofgraph_data = [json.loads(line) for line in f]

    augmented_data = []

    # Initialize GraphProcessor
    graph_processor = GraphProcessor()


    model_id = "./models/Llama-3.1-8B-Instruct"
    pipeline = load_pipeline(model_id)

  # Augment data in batches
    j = 0
    print("Augmenting graph-text pairs in batches ...")
    for i in tqdm(range(0, len(wikiofgraph_data), batch_size)):
        batch = wikiofgraph_data[i:i + batch_size]
        augmented_batch = process_batch(batch, m, n, pipeline)
        augmented_data.extend(augmented_batch)

        # Remove this to augmenting entire dataset
        if j == 1:
            break
        j+=1

    print(f"Original dataset size: {len(wikiofgraph_data)}")
    print(f"Augmented dataset size: {len(augmented_data)}")

    # Initialize a list to store processed items with graphs
    processed_data = []

    print("Generating graphs for augmented data ...")
    # i = 0
    for item in tqdm(augmented_data):
        # print(item)
        # i += 1
        # if i == 7:
        #     break
        try:
            # Extract triples and text
            # triples = item['triplet'].strip('()').split('), ')
            triples = item['triplet']
            text = item['text']

            # Generate graphs using GraphProcessor
            graph, original_triples = graph_processor.create_graph_from_triples(triples)

            # Create a processed item
            processed_item = {
                'input': original_triples,  # Store the original triples for input
                'label': text,             # Store the corresponding text as the label
                'graphs': [graph],         # Store the generated graph
            }
            processed_data.append(processed_item)
        except Exception as e:
            print(f"Error generating graph for item: {e}")
            continue

    print(f"Processed data length with graphs: {len(processed_data)}")

    # Save processed data with graphs
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'processed_WikiofGraph_with_graphs.jsonl')
    with open(output_path, 'w') as f:
        for pair in processed_data:
            f.write(json.dumps(pair, default=str) + '\n')

    print(f"Processed data with graphs saved to {output_path}")

    
    
if __name__ == "__main__":
    main()