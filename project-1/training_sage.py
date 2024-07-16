import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import SAGEConv, to_hetero, HeteroConv
from torch_geometric.loader import LinkNeighborLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score




# create output directories 
os.makedirs("output_embeddings", exist_ok=True)
os.makedirs("output_plots", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# set up logging
log_filename = f"logs/training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


# 1. Graph Creation
def create_graph(interaction_data, ad_features, device):
    logging.info("Starting graph creation...")
    graph = HeteroData()

    # 1. Clean and prepare interaction data
    interaction_data['product_id'] = pd.to_numeric(interaction_data['product_id'], errors='coerce')
    interaction_data = interaction_data.dropna(subset=['product_id', 'client_id'])
    interaction_data['product_id'] = interaction_data['product_id'].astype('int64')
    logging.info("Interaction data shape: {interaction_data.shape}")

    # 2. Process user data
    user_features = interaction_data.groupby('client_id').agg({
        'event_type': lambda x: x.value_counts().get('click', 0) / len(x),  # get click_rate
        'session_id': 'nunique',  # session_count
        'product_id': 'count',  # total_interactions
        'device_category': lambda x: x.mode().iloc[0] if not x.empty else 'unknown',
        'kategori_1': lambda x: x.mode().iloc[0] if not x.empty else 'unknown'
    }).reset_index()

    # Encode categorical features
    for col in ['device_category', 'kategori_1']:
        user_features[col] = pd.Categorical(user_features[col]).codes

    # Normalize user features
    user_features.iloc[:, 1:] = (user_features.iloc[:, 1:] - user_features.iloc[:, 1:].mean()) / user_features.iloc[:, 1:].std()
    
    graph['user'].x = torch.tensor(user_features.iloc[:, 1:].values, dtype=torch.float).to(device)
    user_mapping = {user: idx for idx, user in enumerate(user_features['client_id'])}
    logging.info(f"Number of users: {len(user_mapping)}")

    # 3. Process ad data
    ad_feature_columns = ['kategori_1_x', 'kategori_2_x', 'm2_gross_category', 'room_category', 
                          'floor_category', 'ilan_fiyat_category', 'listing_popularity']
    
    ad_data = ad_features[['id', 'il_x', 'ilce_x', 'mahalle_x'] + ad_feature_columns].copy()
    logging.info(f"Ad data shape: {ad_data.shape}")

    # Handle categorical columns
    for col in ad_feature_columns:
        if col != 'listing_popularity':
            ad_data[col] = pd.Categorical(ad_data[col].fillna('Unknown')).codes
        else:
            ad_data[col] = pd.to_numeric(ad_data[col], errors='coerce').fillna(0)

    # Normalize ad features
    ad_features_normalized = (ad_data[ad_feature_columns] - ad_data[ad_feature_columns].mean()) / ad_data[ad_feature_columns].std()
    logging.info(ad_features_normalized.head())

    # 4. Create location data
    for col in ['il_x', 'ilce_x', 'mahalle_x']:
        ad_data[col] = pd.Categorical(ad_data[col].fillna('Unknown')).codes

    # 5. Create embeddings for location features
    il_embedding = nn.Embedding(ad_data['il_x'].max() + 1, 16).to(device)
    ilce_embedding = nn.Embedding(ad_data['ilce_x'].max() + 1, 16).to(device)
    mahalle_embedding = nn.Embedding(ad_data['mahalle_x'].max() + 1, 16).to(device)

    il_indices = torch.tensor(ad_data['il_x'].values, dtype=torch.long, device=device)
    ilce_indices = torch.tensor(ad_data['ilce_x'].values, dtype=torch.long, device=device)
    mahalle_indices = torch.tensor(ad_data['mahalle_x'].values, dtype=torch.long, device=device)

    il_embeds = il_embedding(il_indices)
    ilce_embeds = ilce_embedding(ilce_indices)
    mahalle_embeds = mahalle_embedding(mahalle_indices)

    # print(f"Location embedding shapes: il {il_embeds.shape}, ilce {ilce_embeds.shape}, mahalle {mahalle_embeds.shape}")

    # 6. Map ads to locations
    location_data = ad_data[['il_x', 'ilce_x', 'mahalle_x']].drop_duplicates()
    location_data['location_id'] = range(len(location_data))
    
    graph['location'].x = torch.cat([
        il_embedding(torch.tensor(location_data['il_x'].values, dtype=torch.long, device=device)),
        ilce_embedding(torch.tensor(location_data['ilce_x'].values, dtype=torch.long, device=device)),
        mahalle_embedding(torch.tensor(location_data['mahalle_x'].values, dtype=torch.long, device=device))
    ], dim=1)
    

    ad_to_location = pd.merge(ad_data[['id', 'il_x', 'ilce_x', 'mahalle_x']], location_data, on=['il_x', 'ilce_x', 'mahalle_x'])
    ad_to_location_dict = dict(zip(ad_to_location['id'], ad_to_location['location_id']))
    
    # Add ad_to_location_dict as an attribute to the graph object
    graph.ad_to_location = ad_to_location_dict



    # 7. Combine ad features with location embeddings
    ad_features_combined = torch.cat([
        torch.tensor(ad_features_normalized.values, dtype=torch.float, device=device),
        il_embeds,
        ilce_embeds,
        mahalle_embeds
    ], dim=1)

    graph['ad'].x = ad_features_combined
    ad_mapping = {ad: idx for idx, ad in enumerate(ad_data['id'])}
    logging.info(f"Number of ads: {len(ad_mapping)}")

    # 8. Create edges in the graph
    ad_indices = torch.tensor([ad_mapping[ad] for ad in ad_to_location['id']], dtype=torch.long, device=device)
    location_indices = torch.tensor(ad_to_location['location_id'].values, dtype=torch.long, device=device)
    graph['ad', 'in', 'location'].edge_index = torch.stack([ad_indices, location_indices])
    graph['location', 'contains', 'ad'].edge_index = torch.stack([location_indices, ad_indices])

    valid_interactions = interaction_data[
        interaction_data['client_id'].isin(user_mapping) & 
        interaction_data['product_id'].isin(ad_mapping)
    ]

    for event_type in ['click', 'purchase']:
        interactions = valid_interactions[valid_interactions['event_type'] == event_type]
        user_indices = torch.tensor([user_mapping[user] for user in interactions['client_id']], dtype=torch.long, device=device)
        ad_indices = torch.tensor([ad_mapping[ad] for ad in interactions['product_id']], dtype=torch.long, device=device)
        edge_index = torch.stack([user_indices, ad_indices])
        graph[('user', event_type, 'ad')].edge_index = edge_index
        graph[('ad', f'rev_{event_type}', 'user')].edge_index = torch.stack([ad_indices, user_indices])

    logging.info("Graph edge types:", graph.edge_types)
    for edge_type in graph.edge_types:
        logging.info(f"Number of {edge_type} edges: {graph[edge_type].edge_index.size(1)}")

    return graph, il_embedding, ilce_embedding, mahalle_embedding, 




# Step 2: Define the GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, metadata, graph, il_embedding, ilce_embedding, mahalle_embedding):
        super().__init__()
        
        self.il_embedding = il_embedding
        self.ilce_embedding = ilce_embedding
        self.mahalle_embedding = mahalle_embedding
        
        node_types, edge_types = metadata
        self.node_feature_dims = {
            'user': graph['user'].x.shape[1],
            'ad': graph['ad'].x.shape[1],
            'location': graph['location'].x.shape[1]
        }

        # First layer of convolutions
        self.conv1 = HeteroConv({
            ('user', 'click', 'ad'): SAGEConv((self.node_feature_dims['user'], self.node_feature_dims['ad']), hidden_channels),
            ('user', 'purchase', 'ad'): SAGEConv((self.node_feature_dims['user'], self.node_feature_dims['ad']), hidden_channels),
            ('ad', 'rev_click', 'user'): SAGEConv((self.node_feature_dims['ad'], self.node_feature_dims['user']), hidden_channels),
            ('ad', 'rev_purchase', 'user'): SAGEConv((self.node_feature_dims['ad'], self.node_feature_dims['user']), hidden_channels),
            ('ad', 'in', 'location'): SAGEConv((self.node_feature_dims['ad'], self.node_feature_dims['location']), hidden_channels),
            ('location', 'contains', 'ad'): SAGEConv((self.node_feature_dims['location'], self.node_feature_dims['ad']), hidden_channels),
        }, aggr='mean')

        # Second layer of convolutions
        self.conv2 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in edge_types
        }, aggr='mean')

        # Third layer of convolutions
        self.conv3 = HeteroConv({
            edge_type: SAGEConv(hidden_channels, hidden_channels)
            for edge_type in edge_types
        }, aggr='mean')
        
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin3 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin4 = torch.nn.Linear(hidden_channels, 1)
        
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(self.dropout(x)) for key, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.relu(self.dropout(x)) for key, x in x_dict.items()}
        
        x_dict = self.conv3(x_dict, edge_index_dict)
        x_dict = {key: F.relu(self.dropout(x)) for key, x in x_dict.items()}
        
        x_dict = {key: self.lin1(x) for key, x in x_dict.items()}
        x_dict = {key: F.relu(self.dropout(x)) for key, x in x_dict.items()}
        
        x_dict = {key: self.lin2(x) for key, x in x_dict.items()}
        return x_dict

    def predict_edge(self, user_emb, ad_emb):
        combined = torch.cat([user_emb, ad_emb], dim=-1)
        hidden = F.relu(self.dropout(self.lin3(combined)))
        return torch.sigmoid(self.lin4(hidden)).squeeze(-1)
        

        
# Step 3: Implement training and evaluation
def train(model, train_loader, optimizer, device, graph):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        node_embeddings = model(batch.x_dict, batch.edge_index_dict)
        
        batch_loss = 0

        for edge_type in [('user', 'click', 'ad'), ('user', 'purchase', 'ad')]:
            if edge_type in batch.edge_index_dict:
                edge_index = batch[edge_type].edge_index
                src, dst = edge_index
                
                src_emb = node_embeddings['user'][src]
                dst_emb = node_embeddings['ad'][dst]

                pred = model.predict_edge(src_emb, dst_emb)
                
                pos_loss = F.binary_cross_entropy(pred, torch.ones_like(pred))
                
                neg_src = src[torch.randperm(src.size(0))]
                neg_dst = dst[torch.randperm(dst.size(0))]
                
                neg_src_emb = node_embeddings['user'][neg_src]
                neg_dst_emb = node_embeddings['ad'][neg_dst]
                
                neg_pred = model.predict_edge(neg_src_emb, neg_dst_emb)
                
                neg_loss = F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
                
                loss = pos_loss + neg_loss
                loss.backward(retain_graph=True)  # Add retain_graph=True here if you need to keep the graph
                batch_loss += loss.item()

                # FUrther Investigation
                # print(f"Edge type: {edge_type}")
                # print(f"Positive predictions range: {pred.min().item():.4f} to {pred.max().item():.4f}")
                # print(f"Negative predictions range: {neg_pred.min().item():.4f} to {neg_pred.max().item():.4f}")

        optimizer.step()
        total_loss += batch_loss
    
    return total_loss / len(train_loader)


# Helper functions (additional metrics)
def calculate_ndcg(y_true, y_pred, k=10):
    return ndcg_score(y_true.reshape(1, -1), y_pred.reshape(1, -1), k=k)

def calculate_mrr(y_true, y_pred):
    order = np.argsort(-y_pred)
    y_true = y_true[order]
    rr = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr) / np.sum(y_true)

def calculate_hit_rate(y_true, y_pred, k=10):
    order = np.argsort(-y_pred)
    y_true = y_true[order]
    return np.sum(y_true[:k]) / np.sum(y_true)


def early_stopping(val_losses, patience=15):
    if len(val_losses) < patience:
        return False
    return all(val_losses[-i-1] <= val_losses[-i] for i in range(1, patience))

def test(model, test_loader, device, graph):
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            node_embeddings = model(batch.x_dict, batch.edge_index_dict)
            
            for edge_type in [('user', 'click', 'ad'), ('user', 'purchase', 'ad')]:
                if edge_type in batch.edge_index_dict:
                    edge_index = batch[edge_type].edge_index
                    src, dst = edge_index
                    
                    src_emb = node_embeddings['user'][src]
                    dst_emb = node_embeddings['ad'][dst]

                    pred = model.predict_edge(src_emb, dst_emb)
                    
                    y_pred.extend(pred.cpu().numpy())
                    y_true.extend(np.ones(len(pred)))
                    
                    # Generate negative samples
                    neg_src = src[torch.randperm(src.size(0))]
                    neg_dst = dst[torch.randperm(dst.size(0))]
                    
                    neg_src_emb = node_embeddings['user'][neg_src]
                    neg_dst_emb = node_embeddings['ad'][neg_dst]
                    
                    neg_pred = model.predict_edge(neg_src_emb, neg_dst_emb)
                    
                    y_pred.extend(neg_pred.cpu().numpy())
                    y_true.extend(np.zeros(len(neg_pred)))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    auc_roc = roc_auc_score(y_true, y_pred)
    avg_precision = average_precision_score(y_true, y_pred)
    
    # Calculate additional metrics
    mrr = calculate_mrr(y_true, y_pred)
    hit_rate = calculate_hit_rate(y_true, y_pred, k=10)
    ndcg = calculate_ndcg(y_true, y_pred, k=10)
    
    return auc_roc, avg_precision, mrr, hit_rate, ndcg



def save_performance_plots(train_losses, val_scores, metric_name, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_scores, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title(f'Training Loss and Validation {metric_name}')
    plt.savefig(f"{output_dir}/{metric_name}_plot.png")
    plt.close()

def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model saved to {filepath}")


def save_embeddings(model, graph, device, output_dir):
    model.eval()
    with torch.no_grad():
        node_embeddings = model(graph.x_dict, graph.edge_index_dict)
    
    for node_type, embeddings in node_embeddings.items():
        torch.save(embeddings.cpu(), f"{output_dir}/{node_type}_embeddings.pt")
    logging.info(f"Embeddings saved to {output_dir}")



# Step 4: Run the model
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    logging.info("loading data...")
    interaction_data = pd.read_csv("davranis.csv")
    ad_features = pd.read_csv("ad_features.csv", low_memory=False)

    logging.info("creating graph...")
    graph, il_embedding, ilce_embedding, mahalle_embedding = create_graph(interaction_data, ad_features, device)
    logging.info("graph created successfully!")
    graph = graph.to(device)

    logging.info("graph edge types:", graph.edge_types)
    for edge_type in graph.edge_types:
        logging.info(f"number of {edge_type} edges: {graph[edge_type].edge_index.size(1)}")

    if ('user', 'click', 'ad') in graph.edge_types and graph[('user', 'click', 'ad')].edge_index.size(1) > 0:
        edge_type = ('user', 'click', 'ad')
    elif ('user', 'purchase', 'ad') in graph.edge_types and graph[('user', 'purchase', 'ad')].edge_index.size(1) > 0:
        logging.info("No 'click' edges found in the graph or 'click' edges are empty. Using 'purchase' edges instead.")
        edge_type = ('user', 'purchase', 'ad')
    else:
        raise ValueError("No valid edges found in the graph. Cannot create loader.")

    ad_feature_count = graph['ad'].x.shape[1]
    user_feature_count = graph['user'].x.shape[1]

    logging.info(f"Ad feature count: {ad_feature_count}")
    logging.info(f"User feature count: {user_feature_count}")

    logging.info(graph.metadata())
    metadata = graph.metadata()
    
    model = GNNModel(hidden_channels=128, out_channels=64, metadata=metadata, 
                     graph=graph,  
                     il_embedding=il_embedding, 
                     ilce_embedding=ilce_embedding, 
                     mahalle_embedding=mahalle_embedding).to(device)

    logging.info(model)

    logging.info("Metdata structure:")
    logging.info(f"Node types: {metadata[0]}")
    logging.info(f"Edge types: {metadata[1]}")
    logging.info(f"User features shape: {graph['user'].x.shape}")
    logging.info(f"Ad features shape: {graph['ad'].x.shape}")

    logging.info("GRaph structure:")
    for node_type in graph.node_types:
        logging.info(f"{node_type} nodes: {graph[node_type].num_nodes}")
    for edge_type in graph.edge_types:
        logging.info(f"{edge_type} edges: {graph[edge_type].num_edges}")

    # split the data into train, validation, and test
    num_edges = graph[edge_type].edge_index.size(1)
    train_ratio, val_ratio = 0.7, 0.15
    train_size = int(num_edges * train_ratio)
    val_size = int(num_edges * val_ratio)

    perm = torch.randperm(num_edges)
    train_edges = perm[:train_size]
    val_edges = perm[train_size:train_size+val_size]
    test_edges = perm[train_size+val_size:]

    train_loader = LinkNeighborLoader(
        data=graph,
        num_neighbors={key: [10, 5] for key in graph.edge_types},
        batch_size=256,
        edge_label_index=(edge_type, graph[edge_type].edge_index[:, train_edges]),
        edge_label=None,
        neg_sampling=None,
    )

    val_loader = LinkNeighborLoader(
        data=graph,
        num_neighbors={key: [10, 5] for key in graph.edge_types},
        batch_size=256,
        edge_label_index=(edge_type, graph[edge_type].edge_index[:, val_edges]),
        edge_label=None,
        neg_sampling=None,
    )

    test_loader = LinkNeighborLoader(
        data=graph,
        num_neighbors={key: [10, 5] for key in graph.edge_types},
        batch_size=256,
        edge_label_index=(edge_type, graph[edge_type].edge_index[:, test_edges]),
        edge_label=None,
        neg_sampling=None,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    val_losses = []
    train_losses = []
    val_auc_rocs = []
    val_avg_precisions = []
    best_auc_roc = 0
    best_model = None

    for epoch in range(10):  
        train_loss = train(model, train_loader, optimizer, device, graph)
        val_auc_roc, val_avg_precision, val_mrr, val_hit_rate, val_ndcg = test(model, val_loader, device, graph)
        
        val_losses.append(val_auc_roc)  
        train_losses.append(train_loss)
        val_auc_rocs.append(val_auc_roc)
        val_avg_precisions.append(val_avg_precision)
        
        scheduler.step(val_auc_roc)
        logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val AUC-ROC: {val_auc_roc:.4f}, '
                    f'Val Avg Precision: {val_avg_precision:.4f}, Val MRR: {val_mrr:.4f}, '
                    f'Val Hit Rate@10: {val_hit_rate:.4f}, Val NDCG@10: {val_ndcg:.4f}')

    # Final test
    test_auc_roc, test_avg_precision, test_mrr, test_hit_rate, test_ndcg = test(model, test_loader, device, graph)
    logging.info(f'Test AUC-ROC: {test_auc_roc:.4f}, Test Avg Precision: {test_avg_precision:.4f}, '
                f'Test MRR: {test_mrr:.4f}, Test Hit Rate@10: {test_hit_rate:.4f}, '
                f'Test NDCG@10: {test_ndcg:.4f}')


    # Save best model
    if best_model is not None:
        model.load_state_dict(best_model)
        save_model(model, "output_models/best_model.pth")

    # Save embeddings
    save_embeddings(model, graph, device, "output_embeddings")

    # Save performance plots
    save_performance_plots(train_losses, val_auc_rocs, 'AUC-ROC', 'output_plots')
    save_performance_plots(train_losses, val_avg_precisions, 'Average Precision', 'output_plots')

if __name__ == '__main__':
    main()

