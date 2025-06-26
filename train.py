import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import k_hop_subgraph
from collections import defaultdict
import random # Added for negative sampling

from model import UserInteractionPredictor 
from data_utils import load_interactions_from_csv, create_weekly_heterodata_from_df

# Hyperparameters
DATA_FILE = 'user_event_interactions.csv'
NUM_WEEKS_TOTAL = 9
TRAIN_WEEKS = 8 

USER_FEATURE_DIM = 32
EVENT_FEATURE_DIM = 64 
SAGE_GNN_HIDDEN_DIM = 128
SAGE_NUM_GNN_LAYERS = 2
SAGE_AGGR = 'mean' 
SAGE_DROPOUT_RATE = 0.1

TRANSFORMER_D_MODEL = SAGE_GNN_HIDDEN_DIM 
TRANSFORMER_NHEAD = 4
TRANSFORMER_NUM_LAYERS = 2 
TRANSFORMER_DROPOUT = 0.1
HISTORICAL_WINDOW_SIZE = 4  # Default window size for transformer
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 256 # Batch size for LinkNeighborLoader
NEIGHBORHOOD_SAMPLING_SIZES = [15, 10] # For 2 GNN layers: 15 neighbors for 1st hop, 10 for 2nd

TOP_K = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def print_model_architecture(model):
    """
    Print the model architecture in a hierarchical format.
    """
    def count_parameters(module):
        try:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)
        except ValueError:
            return "Not initialized yet"
    
    print("\nModel Architecture:")
    print("=" * 50)
    
    # GNN Encoder
    print("\n1. GNN Encoder:")
    print("-" * 30)
    print(f"  User Initial Projection: {model.gnn_encoder.user_initial_proj}")
    print(f"  Event Initial Projection: {model.gnn_encoder.event_initial_proj}")
    print("\n  SAGEConv Layers:")
    for i, (user_conv, event_conv) in enumerate(zip(model.gnn_encoder.user_convs, model.gnn_encoder.event_convs)):
        print(f"    Layer {i+1}:")
        print(f"      User Conv: {user_conv}")
        print(f"      Event Conv: {event_conv}")
        print(f"      User BatchNorm: {model.gnn_encoder.user_bns[i]}")
        print(f"      Event BatchNorm: {model.gnn_encoder.event_bns[i]}")
    
    # Transformer Components
    print("\n2. Transformer Components:")
    print("-" * 30)
    print(f"  GNN to Transformer Projection: {model.gnn_to_transformer_proj}")
    print(f"  Positional Encoding: {model.pos_encoder}")
    print("\n  Transformer Encoder:")
    for i, layer in enumerate(model.transformer_encoder.layers):
        print(f"    Layer {i+1}:")
        print(f"      Self Attention: {layer.self_attn}")
        print(f"      Feed Forward: {layer.linear1} -> {layer.linear2}")
        print(f"      Layer Norm 1: {layer.norm1}")
        print(f"      Layer Norm 2: {layer.norm2}")
    
    # Prediction Head
    print("\n3. Prediction Head:")
    print("-" * 30)
    print(f"  User to Event Scores MLP: {model.user_to_event_scores_mlp}")
    
    # Parameter Counts
    print("\nParameter Counts:")
    print("-" * 30)
    print(f"  GNN Encoder Parameters: {count_parameters(model.gnn_encoder)}")
    print(f"  Transformer Parameters: {count_parameters(model.transformer_encoder)}")
    print(f"  Prediction Head Parameters: {count_parameters(model.user_to_event_scores_mlp)}")
    print(f"  Total Trainable Parameters: {count_parameters(model)}")
    print("=" * 50)
    
    # Print model configuration
    print("\nModel Configuration:")
    print("-" * 30)
    print(f"  User Feature Dimension: {model.gnn_encoder.user_initial_proj.in_features if model.gnn_encoder.user_initial_proj else 'Same as hidden'}")
    print(f"  Event Feature Dimension: {model.gnn_encoder.event_initial_proj.in_features if model.gnn_encoder.event_initial_proj else 'Same as hidden'}")
    print(f"  GNN Hidden Dimension: {model.gnn_encoder.user_convs[0].out_channels}")
    print(f"  Number of GNN Layers: {len(model.gnn_encoder.user_convs)}")
    print(f"  Transformer Model Dimension: {model.transformer_d_model}")
    print(f"  Number of Transformer Layers: {len(model.transformer_encoder.layers)}")
    print(f"  Number of Attention Heads: {model.transformer_encoder.layers[0].self_attn.num_heads}")
    print(f"  Number of Global Events: {model.num_global_events}")
    print("=" * 50)

def main():
    print("Loading interaction data...")
    df_interactions, global_user_map, global_event_map, user_feature_info, event_feature_info = load_interactions_from_csv(
        DATA_FILE,
        static_user_feature_dim=USER_FEATURE_DIM,
        static_event_feature_dim=EVENT_FEATURE_DIM
    )
    NUM_UNIQUE_USERS = user_feature_info['num_unique']
    NUM_GLOBAL_EVENTS = event_feature_info['num_unique']
    print(f"Loaded {len(df_interactions)} interactions.")
    print(f"Number of unique users: {NUM_UNIQUE_USERS}")
    print(f"Number of unique events (pixel_event_id): {NUM_GLOBAL_EVENTS}")

    print("Creating weekly HeteroData snapshots...")
    weekly_data = []
    for week in range(1, NUM_WEEKS_TOTAL + 1):
        weekly_df = df_interactions[df_interactions['week_number'] == week]
        # Pass feature info directly, no need for global maps anymore
        snapshot = create_weekly_heterodata_from_df(weekly_df, 
                                                    user_feature_info['static_features'], 
                                                    event_feature_info['static_features'])
        weekly_data.append(snapshot)
    print(f"Created {len(weekly_data)} weekly snapshots.")

    # --- Training Setup with LinkNeighborLoader ---
    num_train_snapshots = TRAIN_WEEKS
    if num_train_snapshots <= 0:
        print("TRAIN_WEEKS is 0. No training will be performed.")
        train_loader = None
    else:
        print("Combining training snapshots for sampling-based training...")
        # Combine all training graphs into one large graph
        train_snapshots = weekly_data[:num_train_snapshots]
        
        # We need to correctly combine edge indices
        combined_edge_index = torch.cat([s['user', 'interacts_with', 'event'].edge_index for s in train_snapshots], dim=1)
        
        # Create a single data object for the entire training period
        train_data = HeteroData()
        train_data['user'].x = user_feature_info['static_features']
        train_data['event'].x = event_feature_info['static_features']
        train_data['user', 'interacts_with', 'event'].edge_index = combined_edge_index
        train_data = ToUndirected()(train_data) # Ensure reverse edges are created
        
        print("Setting up LinkNeighborLoader for training...")
        train_loader = LinkNeighborLoader(
            train_data,
            num_neighbors=NEIGHBORHOOD_SAMPLING_SIZES,
            batch_size=BATCH_SIZE,
            edge_label_index=(('user', 'interacts_with', 'event'), train_data['user', 'interacts_with', 'event'].edge_index),
            neg_sampling_ratio=1.0, # 1 negative sample per positive sample
            shuffle=True
        )

    # --- Evaluation Setup ---
    eval_target_snapshot_idx = num_train_snapshots
    perform_evaluation = (eval_target_snapshot_idx < len(weekly_data)) and (eval_target_snapshot_idx >= HISTORICAL_WINDOW_SIZE)

    print("Initializing model...")
    model = UserInteractionPredictor(
        user_feature_dim=USER_FEATURE_DIM,
        event_feature_dim=EVENT_FEATURE_DIM,
        gnn_hidden_dim=SAGE_GNN_HIDDEN_DIM,
        num_gnn_layers=SAGE_NUM_GNN_LAYERS,
        sage_aggr=SAGE_AGGR,
        sage_dropout_rate=SAGE_DROPOUT_RATE,
        transformer_d_model=TRANSFORMER_D_MODEL,
        transformer_nhead=TRANSFORMER_NHEAD,
        transformer_num_layers=TRANSFORMER_NUM_LAYERS,
        transformer_dropout=TRANSFORMER_DROPOUT,
        num_unique_users=NUM_UNIQUE_USERS,
        num_global_events=NUM_GLOBAL_EVENTS
    ).to(device)
    
    print_model_architecture(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    if train_loader:
        print("Starting training with neighborhood sampling...")
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            
            for batch in train_loader:
                batch = batch.to(device)
                
                # Get GNN embeddings for nodes in the sampled subgraph
                user_gnn_embs, event_gnn_embs = model.gnn_encoder(batch.x_dict, batch.edge_index_dict)
                
                # The Transformer is NOT used in this sampling-based training loop.
                # We directly use the GNN output for link prediction.
                # We need to map the batch's node indices to the global indices for the prediction head.
                pos_user_indices = batch['user', 'interacts_with', 'event'].edge_label_index[0]
                pos_event_indices = batch['user', 'interacts_with', 'event'].edge_label_index[1]

                # Negative samples are provided in `edge_label_index` after positive ones
                num_pos = pos_user_indices.size(0)
                neg_user_indices = batch['user', 'interacts_with', 'event'].edge_label_index[0, num_pos:]
                neg_event_indices = batch['user', 'interacts_with', 'event'].edge_label_index[1, num_pos:]
                
                # The model's forward pass for training now takes GNN embeddings directly
                model_output = model(
                    user_gnn_embs, event_gnn_embs,
                    {'user': pos_user_indices, 'event': pos_event_indices},
                    {'user': neg_user_indices, 'event': neg_event_indices}
                )

                pos_labels = torch.ones_like(model_output['positive_scores'])
                neg_labels = torch.zeros_like(model_output['negative_scores'])

                all_scores = torch.cat([model_output['positive_scores'], model_output['negative_scores']], dim=0)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                if all_scores.numel() > 0:
                    loss = criterion(all_scores, all_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item() * BATCH_SIZE

            avg_loss = total_loss / (len(train_loader.dataset) if hasattr(train_loader, 'dataset') else 1)
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Loss: {avg_loss:.4f}")

    # --- Evaluation Logic ---
    if perform_evaluation:
        print("\nStarting evaluation...")
        model.eval()
        
        eval_hist_start_idx = eval_target_snapshot_idx - HISTORICAL_WINDOW_SIZE
        eval_input_snapshots = [s.to(device) for s in weekly_data[eval_hist_start_idx : eval_target_snapshot_idx + 1]]
        
        # For evaluation, we use the original temporal pipeline
        # Get temporal embeddings using the GNN's inference mode
        user_temporal_embs, event_temporal_embs = model.get_temporal_embeddings(eval_input_snapshots, device)

        # Get actual interactions from the target week for evaluation metrics
        target_snapshot_eval = weekly_data[eval_target_snapshot_idx]
        actual_interactions = defaultdict(set)
        for u, e in target_snapshot_eval['user', 'interacts_with', 'event'].edge_index.t().tolist():
            actual_interactions[u].add(e)
            
        if not actual_interactions:
            print("No actual interactions in evaluation week. Skipping metrics.")
            return

        total_recall = 0
        total_precision = 0
        evaluated_users = 0

        for user_id, true_events in actual_interactions.items():
            if not true_events:
                continue

            user_embedding = user_temporal_embs[user_id]
            
            # Predict top-K events for this user
            pred_event_indices, _ = model.predict_top_k_for_user(user_embedding, event_temporal_embs, k=TOP_K)
            
            predicted_set = set(pred_event_indices)
            true_set = true_events
            
            # Calculate precision and recall for this user
            true_positives = len(predicted_set.intersection(true_set))
            if len(predicted_set) > 0:
                precision = true_positives / len(predicted_set)
            else:
                precision = 0
            
            if len(true_set) > 0:
                recall = true_positives / len(true_set)
            else:
                recall = 0
                
            total_precision += precision
            total_recall += recall
            evaluated_users += 1
        
        if evaluated_users > 0:
            avg_precision = total_precision / evaluated_users
            avg_recall = total_recall / evaluated_users
            print(f"\nEvaluation Results for Week {eval_target_snapshot_idx + 1}:")
            print(f"  - Average Precision@{TOP_K}: {avg_precision:.4f}")
            print(f"  - Average Recall@{TOP_K}: {avg_recall:.4f}")
        else:
            print("No users with interactions to evaluate.")

    print("\nScript finished.")
    print(f"Reminder: Current torch version: {torch.__version__}. Ensure compatibility with PyG.")

if __name__ == '__main__':
    main() 