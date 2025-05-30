import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
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
NUM_NEGATIVE_SAMPLES_PER_POSITIVE = 1

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
        if weekly_df.empty and (NUM_UNIQUE_USERS > 0 and NUM_GLOBAL_EVENTS > 0):
            snapshot = HeteroData()
            snapshot['user'].x = user_feature_info['static_features'].clone().detach()
            snapshot['event'].x = event_feature_info['static_features'].clone().detach()
            snapshot['user', 'interacts_with', 'event'].edge_index = torch.empty((2,0), dtype=torch.long)
            snapshot['user'].num_nodes = NUM_UNIQUE_USERS
            snapshot['event'].num_nodes = NUM_GLOBAL_EVENTS
        elif not weekly_df.empty:
            snapshot = create_weekly_heterodata_from_df(weekly_df, global_user_map, global_event_map, 
                                                        user_feature_info['static_features'], 
                                                        event_feature_info['static_features'])
        else:
            continue
        weekly_data.append(snapshot)
    
    if not weekly_data:
        print("No weekly data snapshots were created. Exiting.")
        return
    print(f"Created {len(weekly_data)} weekly snapshots.")

    num_total_snapshots = len(weekly_data)
    num_train_snapshots = TRAIN_WEEKS 

    if num_train_snapshots == 0:
        print("TRAIN_WEEKS is 0. No training will be performed. Skipping to evaluation if possible.")
    elif num_train_snapshots <= HISTORICAL_WINDOW_SIZE:
        raise ValueError(f"TRAIN_WEEKS ({TRAIN_WEEKS}) must be > HISTORICAL_WINDOW_SIZE ({HISTORICAL_WINDOW_SIZE})")
    elif num_train_snapshots < 1:
         raise ValueError(f"TRAIN_WEEKS ({TRAIN_WEEKS}) must be at least 1.")

    weekly_data_train = weekly_data[:num_train_snapshots]
    eval_target_snapshot_idx = num_train_snapshots 
    perform_evaluation = False
    if eval_target_snapshot_idx < num_total_snapshots:
        if eval_target_snapshot_idx < HISTORICAL_WINDOW_SIZE:
            print("Not enough historical data for evaluation. Skipping evaluation.")
        else:
            perform_evaluation = True
    else:
        print("Not enough data for the designated evaluation week. Skipping evaluation.")

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
    
    # Print model architecture
    print_model_architecture(model)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    if num_train_snapshots > 0:
        print("Starting training...")
        
        # Training loop: predicts snapshot t+1 using history up to t
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            num_batches_processed = 0
            
            for t in range(HISTORICAL_WINDOW_SIZE - 1, num_train_snapshots - 1):
                target_snapshot_data_train = weekly_data_train[t+1].to(device)
                
                positive_edge_index = target_snapshot_data_train['user', 'interacts_with', 'event'].edge_index
                target_positive_user_indices = positive_edge_index[0]
                target_positive_event_indices = positive_edge_index[1]

                if target_positive_user_indices.numel() == 0:
                    continue

                num_positive_interactions = target_positive_user_indices.size(0)
                user_to_positive_events_target = {i: set() for i in range(NUM_UNIQUE_USERS)}
                for i in range(num_positive_interactions):
                    user_to_positive_events_target[target_positive_user_indices[i].item()].add(target_positive_event_indices[i].item())

                batch_user_indices_for_negatives = []
                batch_negative_event_indices = []

                for i in range(num_positive_interactions):
                    user_idx = target_positive_user_indices[i].item()
                    for _ in range(NUM_NEGATIVE_SAMPLES_PER_POSITIVE):
                        num_sampling_attempts = 0
                        while True:
                            neg_event_idx = random.randint(0, NUM_GLOBAL_EVENTS - 1)
                            if neg_event_idx not in user_to_positive_events_target[user_idx]:
                                batch_user_indices_for_negatives.append(user_idx)
                                batch_negative_event_indices.append(neg_event_idx)
                                break
                            num_sampling_attempts += 1
                            if num_sampling_attempts > NUM_GLOBAL_EVENTS * 2:
                                break 
                
                if not batch_user_indices_for_negatives:
                    continue

                user_indices_for_negatives_tensor = torch.tensor(batch_user_indices_for_negatives, dtype=torch.long, device=device)
                negative_event_indices_for_loss_tensor = torch.tensor(batch_negative_event_indices, dtype=torch.long, device=device)

                hist_start_idx = (t + 1) - HISTORICAL_WINDOW_SIZE
                model_input_snapshots = [snap.to(device) for snap in weekly_data_train[hist_start_idx : t+2]]
                
                model_output = model(
                    model_input_snapshots,
                    {'user': target_positive_user_indices, 'event': target_positive_event_indices},
                    {'user': user_indices_for_negatives_tensor, 'event': negative_event_indices_for_loss_tensor}
                )

                pos_labels = torch.ones_like(model_output['positive_scores'], device=device)
                neg_labels = torch.zeros_like(model_output['negative_scores'], device=device)

                all_scores = torch.cat([model_output['positive_scores'], model_output['negative_scores']], dim=0)
                all_labels = torch.cat([pos_labels, neg_labels], dim=0)

                if all_scores.numel() > 0:
                    loss = criterion(all_scores, all_labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    num_batches_processed += 1
            
            avg_epoch_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Average Training Loss: {avg_epoch_loss:.4f}")
    else:
        print("Skipping training phase as TRAIN_WEEKS is 0 or insufficient.")

    if perform_evaluation:
        print("\nStarting evaluation...")
        model.eval()
        all_precisions = []
        all_recalls = []

        eval_hist_start_idx = eval_target_snapshot_idx - HISTORICAL_WINDOW_SIZE
        
        if eval_hist_start_idx < 0:
            print(f"Warning: eval_hist_start_idx ({eval_hist_start_idx}) is < 0. Using 0. Less history will be used.")
            eval_hist_start_idx = 0
        
        if eval_target_snapshot_idx < 0:
            print("Evaluation target index is invalid. Skipping evaluation.")
        else:
            model_input_snapshots_eval = [snap.to(device) for snap in weekly_data[eval_hist_start_idx : eval_target_snapshot_idx + 1]]
            weekly_data_eval_target_actual = weekly_data[eval_target_snapshot_idx].to(device)

            with torch.no_grad():
                all_users_temporal_embeddings_eval, all_events_embeddings_eval = model.get_temporal_embeddings(model_input_snapshots_eval)

                eval_ground_truth_edges = weekly_data_eval_target_actual['user', 'interacts_with', 'event'].edge_index
                user_to_true_events_eval = {i: set() for i in range(NUM_UNIQUE_USERS)}
                for i in range(eval_ground_truth_edges[0].size(0)):
                    user_to_true_events_eval[eval_ground_truth_edges[0][i].item()].add(eval_ground_truth_edges[1][i].item())

                unique_users_in_eval_week = torch.unique(eval_ground_truth_edges[0]).tolist()
                if not unique_users_in_eval_week:
                    print("No users with interactions in the evaluation week. Skipping P@k/R@k calculation.")
                else:
                    for user_id_eval in unique_users_in_eval_week:
                        true_events_for_user = user_to_true_events_eval.get(user_id_eval)
                        if not true_events_for_user:
                            continue 

                        user_temporal_embedding_eval = all_users_temporal_embeddings_eval[user_id_eval]
                        predicted_top_k_indices, _ = model.predict_top_k_for_user(
                            user_temporal_embedding_eval,
                            all_events_embeddings_eval,
                            k=TOP_K
                        )
                        
                        true_positives = set(predicted_top_k_indices) & true_events_for_user
                        
                        precision = len(true_positives) / TOP_K if TOP_K > 0 else 0
                        recall = len(true_positives) / len(true_events_for_user) if len(true_events_for_user) > 0 else 0
                        
                        all_precisions.append(precision)
                        all_recalls.append(recall)

                    if all_precisions and all_recalls:
                        avg_precision = np.mean(all_precisions)
                        avg_recall = np.mean(all_recalls)
                        print(f"Evaluation Results (on week {eval_target_snapshot_idx + 1}):")
                        print(f"  Average Precision@{TOP_K}: {avg_precision:.4f}")
                        print(f"  Average Recall@{TOP_K}: {avg_recall:.4f}")
                    else:
                        print("No evaluation metrics calculated.")                    
    else:
        print("Skipping evaluation.")

    print("\nScript finished.")
    print(f"Reminder: Current torch version: {torch.__version__}. Ensure compatibility with PyG.")

if __name__ == '__main__':
    main() 