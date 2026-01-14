import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "gnn_dataset.pt")
OUTPUT_DIR = BASE_DIR
BATCH_SIZE = 32
HIDDEN_DIM = 64
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class GNNDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    """
    Collate function to create a disjoint union graph batch.
    """
    x_list = []
    edge_index_list = []
    y_list = []
    batch_vec_list = []
    
    node_offset = 0
    
    for i, data in enumerate(batch):
        x = data['x']
        edge_index = data['edge_index']
        y = data['y']
        
        num_nodes = x.size(0)
        
        # Features
        x_list.append(x)
        
        # Edges (shifted by offset)
        if edge_index.size(1) > 0:
            edge_index_shifted = edge_index + node_offset
            edge_index_list.append(edge_index_shifted)
            
        # Label
        y_list.append(y)
        
        # Batch vector (to identify which graph a node belongs to)
        batch_vec_list.append(torch.full((num_nodes,), i, dtype=torch.long))
        
        node_offset += num_nodes
        
    # Concatenate
    x_batch = torch.cat(x_list, dim=0)
    batch_vec = torch.cat(batch_vec_list, dim=0)
    y_batch = torch.cat(y_list, dim=0)
    
    if edge_index_list:
        edge_index_batch = torch.cat(edge_index_list, dim=1)
    else:
        edge_index_batch = torch.zeros((2, 0), dtype=torch.long)
        
    return x_batch, edge_index_batch, batch_vec, y_batch

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges]
        
        num_nodes = x.size(0)
        
        # Add self-loops
        indices = torch.arange(num_nodes, device=x.device)
        self_loops = torch.stack([indices, indices], dim=0)
        
        if edge_index.size(1) > 0:
            edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)
        else:
            edge_index_with_loops = self_loops
            
        # Adjacency matrix construction (Sparse)
        row, col = edge_index_with_loops
        deg = torch.bincount(row, minlength=num_nodes).float()
        
        # Compute Normalization
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        # Edge Values
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Sparse Matrix
        adj = torch.sparse_coo_tensor(
            edge_index_with_loops, 
            norm, 
            (num_nodes, num_nodes)
        )
        
        # Message Passing
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        
        return output + self.bias

class VulnerabilityGNN(nn.Module):
    def __init__(self, vocab_size, hidden_dim):
        super(VulnerabilityGNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        self.conv1 = GCNLayer(hidden_dim, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, hidden_dim)
        self.conv3 = GCNLayer(hidden_dim, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)
        
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x, edge_index, batch_vec):
        # 1. Embedding
        h = self.embedding(x)
        
        # 2. GCN Layers
        h = self.conv1(h, edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = self.relu(h)
        h = self.dropout(h)
        
        h = self.conv3(h, edge_index)
        h = self.relu(h)
        
        # 3. Global Pooling (Readout) - Mean Pooling
        batch_size = batch_vec.max().item() + 1
        h_graph = torch.zeros(batch_size, h.size(1), device=h.device)
        counts = torch.zeros(batch_size, 1, device=h.device)
        
        h_graph.index_add_(0, batch_vec, h)
        
        ones = torch.ones(h.size(0), 1, device=h.device)
        # Using index_add_ for scatter sum of counts
        counts.index_add_(0, batch_vec, ones)
        
        h_graph = h_graph / (counts + 1e-6)
        
        # 4. Classifier
        out = self.fc1(h_graph)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

def train_model():
    print("Loading dataset...")
    saved_data = torch.load(DATASET_PATH)
    dataset_list = saved_data['dataset']
    vocab = saved_data['vocab']
    
    # Calculate counts
    total_samples = len(dataset_list)
    pos_samples = sum([d['y'].item() for d in dataset_list])
    neg_samples = total_samples - pos_samples
    print(f"Total: {total_samples}, Pos: {pos_samples:g}, Neg: {neg_samples:g}")
    
    # Class weight
    pos_weight = torch.tensor([neg_samples / (pos_samples + 1e-6)], device=DEVICE)
    print(f"Using pos_weight: {pos_weight.item():.2f}")
    
    print(f"Loaded {total_samples} samples. Vocab size: {len(vocab)}")
    
    # Check if we have positive samples
    if pos_samples == 0:
        print("WARNING: No positive samples found! Weighted loss will be huge/invalid.")
        # Override to 1.0 to prevent crash, but training will be meaningless
        pos_weight = torch.tensor([1.0], device=DEVICE)
        
    # Split
    import random
    random.seed(42)
    random.shuffle(dataset_list)
    
    train_size = int(0.7 * total_samples)
    val_size = int(0.15 * total_samples)
    test_size = total_samples - train_size - val_size
    
    train_data = dataset_list[:train_size]
    val_data = dataset_list[train_size:train_size+val_size]
    test_data = dataset_list[train_size+val_size:]
    
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    train_loader = DataLoader(GNNDataset(train_data), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(GNNDataset(val_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(GNNDataset(test_data), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # Model
    model = VulnerabilityGNN(vocab_size=len(vocab), hidden_dim=HIDDEN_DIM)
    model.to(DEVICE)
    
    # Weighted Loss
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Metrics
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for x, edge_index, batch_vec, y in train_loader:
            x, edge_index, batch_vec, y = x.to(DEVICE), edge_index.to(DEVICE), batch_vec.to(DEVICE), y.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(x, edge_index, batch_vec)
            loss = criterion(output.view(-1), y.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for x, edge_index, batch_vec, y in val_loader:
                x, edge_index, batch_vec, y = x.to(DEVICE), edge_index.to(DEVICE), batch_vec.to(DEVICE), y.to(DEVICE)
                output = model(x, edge_index, batch_vec)
                loss = criterion(output.view(-1), y.view(-1))
                val_loss += loss.item()
                
                preds = (output > 0).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        val_rec = recall_score(all_labels, all_preds, zero_division=0)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val Rec: {val_rec:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_gnn_model.pth'))
            
    print(f"\nTraining complete in {time.time() - start_time:.2f}s")
    
    # Test Evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'best_gnn_model.pth')))
    model.eval()
    
    test_preds = []
    test_probs = []
    test_labels = []
    
    print("\nEvaluating on Test Set...")
    with torch.no_grad():
        for x, edge_index, batch_vec, y in test_loader:
            x, edge_index, batch_vec, y = x.to(DEVICE), edge_index.to(DEVICE), batch_vec.to(DEVICE), y.to(DEVICE)
            output = model(x, edge_index, batch_vec)
            
            test_preds.extend((output > 0).float().cpu().numpy())
            test_probs.extend(torch.sigmoid(output).cpu().numpy())
            test_labels.extend(y.cpu().numpy())
            
    # Calculate Final Metrics
    acc = accuracy_score(test_labels, test_preds)
    prec = precision_score(test_labels, test_preds, zero_division=0)
    rec = recall_score(test_labels, test_preds, zero_division=0)
    f1 = f1_score(test_labels, test_preds, zero_division=0)
    conf_mat = confusion_matrix(test_labels, test_preds)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS (GNN)")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("-" * 30)
    print("Confusion Matrix:")
    print(conf_mat)
    print("="*50)
    
    # Save Report
    with open(os.path.join(OUTPUT_DIR, 'gnn_report.txt'), 'w') as f:
        f.write("GNN VULNERABILITY DETECTION REPORT\n")
        f.write("==================================\n\n")
        f.write(f"Model: Custom GCN (3 layers)\n")
        f.write(f"Hidden Dim: {HIDDEN_DIM}\n")
        f.write(f"Epochs: {EPOCHS}\n\n")
        f.write(f"Weighted Loss: Yes (Pos Weight: {pos_weight.item():.2f})\n\n")
        f.write(f"TEST METRICS:\n")
        f.write(f"Accuracy:  {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1 Score:  {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(conf_mat))

if __name__ == "__main__":
    train_model()
