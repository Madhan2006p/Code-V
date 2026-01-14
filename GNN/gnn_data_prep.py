import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import re

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up to Data/Processed/AST
AST_DIR = os.path.join(os.path.dirname(BASE_DIR), "Data", "Processed", "AST")
OUTPUT_FILE = os.path.join(BASE_DIR, "gnn_dataset.pt")

def parse_ast_file(filepath):
    """
    Parses a single AST file into nodes and edges.
    Format is expected to be tab-separated, with Depth at implicit or explicit column.
    Based on file view: NodeType \t Start \t End \t Depth \t Content...
    """
    nodes = [] # List of node types (strings)
    edges = [] # List of (src, dst) tuples
    stack = [] # Stack of (index, depth)
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            parts = line.strip().split('\t')
            if not parts:
                continue
                
            # Heuristic parsing based on observed format
            # Example: PARAMETER_DECL \t 2:32 \t 2:48 \t 3 \t AVCodecContext * avctx
            # Index 0: Type, Index 3: Depth (usually)
            
            if len(parts) < 4:
                # Fallback or skip
                continue
                
            node_type = parts[0]
            try:
                depth = int(parts[3])
            except ValueError:
                # Try finding an integer in other columns if format varies
                depth = 0
                for p in parts[1:]:
                    if p.isdigit():
                        depth = int(p)
                        break
            
            # Add node
            current_idx = len(nodes)
            nodes.append(node_type)
            
            # Determine parent
            # Stack contains ancestors. 
            # We pop from stack while stack_top.depth >= current_depth
            # The new stack top is the parent.
            
            while stack and stack[-1][1] >= depth:
                stack.pop()
                
            if stack:
                parent_idx = stack[-1][0]
                edges.append((parent_idx, current_idx))
                # Make undirected? Usually GNNs use undirected or add both directions
                edges.append((current_idx, parent_idx))
            
            stack.append((current_idx, depth))
            
    return nodes, edges

def process_dataset():
    files = glob.glob(os.path.join(AST_DIR, "*.txt"))
    print(f"Found {len(files)} AST files.")
    
    dataset = []
    node_type_vocab = {'<PAD>': 0, '<UNK>': 1}
    
    for filepath in tqdm(files, desc="Processing ASTs"):
        filename = os.path.basename(filepath)
        
        # Label logic
        label = 1 if "cve" in filename.lower() else 0
        
        try:
            nodes, edges = parse_ast_file(filepath)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue
            
        if not nodes:
            continue
            
        # Convert node types to indices
        node_indices = []
        for nt in nodes:
            if nt not in node_type_vocab:
                node_type_vocab[nt] = len(node_type_vocab)
            node_indices.append(node_type_vocab[nt])
        
        # Convert to tensors
        x = torch.tensor(node_indices, dtype=torch.long)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Handle single-node graph (no edges)
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            
        data = {
            'x': x,
            'edge_index': edge_index,
            'y': torch.tensor([label], dtype=torch.float),
            'filename': filename
        }
        dataset.append(data)
        
    print(f"Processed {len(dataset)} graphs.")
    print(f"Vocab size: {len(node_type_vocab)}")
    
    # Save
    torch.save({
        'dataset': dataset,
        'vocab': node_type_vocab
    }, OUTPUT_FILE)
    print(f"Saved dataset to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_dataset()
