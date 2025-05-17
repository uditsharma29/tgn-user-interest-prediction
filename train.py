import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd # Added for type hinting if needed, though not strictly necessary for this script

from model import UserInteractionPredictor
from data_utils import load_interactions_from_csv, create_weekly_heterodata_from_df, USER_FEAT_DIM_DEFAULT, ITEM_FEAT_DIM_DEFAULT

# --- Constants & Configuration ---
CSV_FILE_PATH = "user_item_interactions.csv"
HISTORICAL_WINDOW_SIZE = 3
MAX_WEEKS_TOTAL = 9 # Max week number in the dataset
TRAIN_UP_TO_WEEK = 8 # Weeks 1 to 8 for training
# Week 9 will be used for validation/testing in the current setup

# --- Model Hyperparameters (SAGEConv based) ---
GNN_HIDDEN_CHANNELS = 32
SAGE_AGGREGATOR = 'mean'
SAGE_DROPOUT_RATE = 0.1
GNN_MOMENTUM_BN = 0.1
TRANSFORMER_D_MODEL = GNN_HIDDEN_CHANNELS
TR_HEADS = 2
TR_ENCODER_LAYERS = 2
TR_DECODER_LAYERS = 2
TR_DROPOUT = 0.1
TRANSFORMER_MAX_SEQ_LEN = HISTORICAL_WINDOW_SIZE + 5 # Should be >= HISTORICAL_WINDOW_SIZE

# --- Training Hyperparameters ---
LEARNING_RATE = 0.005
EPOCHS = 25 # Slightly increased epochs
TOP_K = 5

# --- Evaluation Helper ---
def calculate_precision_recall_at_k(predicted_top_k, true_positives_for_user, k):
    if not true_positives_for_user:
        return (1.0, 1.0) if not predicted_top_k else (0.0, 0.0)
    predicted_set = set(predicted_top_k)
    hits = len(predicted_set.intersection(true_positives_for_user))
    precision_k = hits / k if k > 0 else 0.0
    recall_k = hits / len(true_positives_for_user) if len(true_positives_for_user) > 0 else 0.0
    return precision_k, recall_k

def main():
    # --- Load Data from CSV ---
    try:
        interactions_df, user_map, item_map, num_unique_users, num_unique_items = \
            load_interactions_from_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"ERROR: Dataset CSV file not found at {CSV_FILE_PATH}.")
        print("Please run `generate_dataset_csv.py` first to create the dataset.")
        return
    
    # Use actual feature dimensions if needed, or stick to defaults for random features
    actual_user_feat_dim = USER_FEAT_DIM_DEFAULT
    actual_item_feat_dim = ITEM_FEAT_DIM_DEFAULT

    # --- Generate Static Node Features (Randomly in this example) ---
    # These are generated once and used for all weekly snapshots.
    static_user_features = torch.randn(num_unique_users, actual_user_feat_dim)
    static_item_features = torch.randn(num_unique_items, actual_item_feat_dim)
    print(f"Generated static random features for {num_unique_users} users and {num_unique_items} items.")

    # --- Create All Weekly HeteroData Snapshots ---
    all_weekly_snapshots = []
    print(f"Generating weekly graph snapshots from week 1 to {MAX_WEEKS_TOTAL}...")
    for week_num in range(1, MAX_WEEKS_TOTAL + 1):
        week_df = interactions_df[interactions_df['week_number'] == week_num]
        snapshot = create_weekly_heterodata_from_df(
            week_df, 
            num_unique_users, 
            num_unique_items,
            static_user_features,
            static_item_features
        )
        all_weekly_snapshots.append(snapshot)
        # print(f"  Snapshot for week {week_num} created. Interactions: {snapshot['user', 'interacts_with', 'item'].edge_index.size(1) if snapshot else 0}")

    print(f"Generated {len(all_weekly_snapshots)} weekly snapshots in total.")
    if not all_weekly_snapshots:
        print("No snapshots generated from CSV data. Exiting.")
        return

    initial_metadata = all_weekly_snapshots[0].metadata()
    print(f"Initial graph metadata (from Week 1 snapshot): {initial_metadata}")

    model = UserInteractionPredictor(
        metadata=initial_metadata,
        user_feat_dim=actual_user_feat_dim, # Raw feature dim fed to GNN for 'user'
        item_feat_dim=actual_item_feat_dim, # Raw feature dim fed to GNN for 'item'
        gnn_hidden_channels=GNN_HIDDEN_CHANNELS,
        sage_aggr=SAGE_AGGREGATOR,
        sage_dropout_rate=SAGE_DROPOUT_RATE,
        gnn_momentum_bn=GNN_MOMENTUM_BN,
        transformer_d_model=TRANSFORMER_D_MODEL,
        tr_heads=TR_HEADS,
        tr_encoder_layers=TR_ENCODER_LAYERS,
        tr_decoder_layers=TR_DECODER_LAYERS,
        tr_dropout=TR_DROPOUT,
        num_global_items=num_unique_items, # Critical for the prediction head
        transformer_max_seq_len=TRANSFORMER_MAX_SEQ_LEN
    )
    print("UserInteractionPredictor model initialized using data from CSV.")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    print("Starting training...")
    # Training loop: uses weeks up to TRAIN_UP_TO_WEEK
    # The target week for prediction will be week `i + HISTORICAL_WINDOW_SIZE + 1` (1-indexed)
    # So, if history is 3 weeks, first target is week 4.
    # Loop goes from week 0 (data for week 1) up to index for (TRAIN_UP_TO_WEEK - HISTORICAL_WINDOW_SIZE -1)
    
    num_training_targets = TRAIN_UP_TO_WEEK - HISTORICAL_WINDOW_SIZE
    if num_training_targets <=0:
        print(f"Not enough weeks for training with window size {HISTORICAL_WINDOW_SIZE} and train up to week {TRAIN_UP_TO_WEEK}.")
        return

    for epoch in range(EPOCHS):
        model.train()
        total_epoch_loss = 0
        num_loss_samples = 0

        # Iterate through possible start points for historical windows for training
        # Snapshot indices are 0-indexed (week 1 is at index 0)
        for i in range(num_training_targets):
            # historical_window snapshots: weeks [i+1, ..., i+HISTORICAL_WINDOW_SIZE]
            # target_snapshot: week i+HISTORICAL_WINDOW_SIZE+1
            history_start_idx = i
            history_end_idx = i + HISTORICAL_WINDOW_SIZE
            target_idx = history_end_idx 

            if target_idx >= len(all_weekly_snapshots) or target_idx >= TRAIN_UP_TO_WEEK:
                continue # Ensure target is within training period and available data

            historical_window = all_weekly_snapshots[history_start_idx : history_end_idx]
            target_snapshot = all_weekly_snapshots[target_idx]
            
            current_target_week_num = target_idx + 1 # 1-indexed week number
            # print(f"Epoch {epoch+1}, Train step {i}: History weeks {[h_idx+1 for h_idx in range(history_start_idx, history_end_idx)]}, Target week {current_target_week_num}")

            # ---- BEGIN: Print statements for studying input to model ----
            if i == 0: # Only print for the first training step of the epoch
                print(f"\n--- Epoch {epoch+1}, First Training Step (Target Week: {current_target_week_num}) --- Model Input Data ---")
                print(f"Historical Window (Weeks {[w_idx+1 for w_idx in range(history_start_idx, history_end_idx)]}):")
                for idx, hist_snap in enumerate(historical_window):
                    print(f"  Historical Snapshot {idx+1} (Week {history_start_idx + idx + 1}):")
                    print(f"    x_dict keys: {list(hist_snap.x_dict.keys())}")
                    for node_type, features in hist_snap.x_dict.items():
                        print(f"      '{node_type}' features shape: {features.shape}")
                    print(f"    edge_index_dict keys: {list(hist_snap.edge_index_dict.keys())}")
                    for edge_type, e_idx in hist_snap.edge_index_dict.items():
                        print(f"      '{edge_type}' edge_index shape: {e_idx.shape}")
                
                print(f"Target Snapshot (Week {current_target_week_num}):")
                print(f"  x_dict keys: {list(target_snapshot.x_dict.keys())}")
                for node_type, features in target_snapshot.x_dict.items():
                    print(f"    '{node_type}' features shape: {features.shape}")
                print(f"  edge_index_dict keys: {list(target_snapshot.edge_index_dict.keys())}")
                for edge_type, e_idx in target_snapshot.edge_index_dict.items():
                    print(f"    '{edge_type}' edge_index shape: {e_idx.shape}")
                if ('user', 'interacts_with', 'item') in target_snapshot.edge_types:
                    print(f"  Target edge_label_index ('user', 'interacts_with', 'item') shape: {target_snapshot['user', 'interacts_with', 'item'].edge_label_index.shape}")
                    print(f"  Target edge_label ('user', 'interacts_with', 'item') shape: {target_snapshot['user', 'interacts_with', 'item'].edge_label.shape}")
                print("--- End Model Input Data ---")
            # ---- END: Print statements for studying input to model ----

            optimizer.zero_grad()
            
            if ('user', 'interacts_with', 'item') not in target_snapshot.edge_types or \
                target_snapshot['user', 'interacts_with', 'item'].edge_label_index.numel() == 0:
                # print(f"  Skipping training for target week {current_target_week_num}, no observed interactions.")
                continue
            
            predicted_scores_for_observed = model(historical_window, target_snapshot)
            true_counts_for_observed = target_snapshot['user', 'interacts_with', 'item'].edge_label

            if predicted_scores_for_observed.shape != true_counts_for_observed.shape or true_counts_for_observed.numel() == 0:
                # print(f"  Shape mismatch or no true labels for target week {current_target_week_num}. Pred: {predicted_scores_for_observed.shape}, True: {true_counts_for_observed.shape}. Skipping.")
                continue

            loss = criterion(predicted_scores_for_observed, true_counts_for_observed)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item() * true_counts_for_observed.size(0)
            num_loss_samples += true_counts_for_observed.size(0)

        avg_epoch_loss = total_epoch_loss / num_loss_samples if num_loss_samples > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS}, Avg Training Loss (on observed interactions): {avg_epoch_loss:.4f}")

        # --- Evaluation for Top-K (using Week 9 as validation/test) ---
        if (epoch + 1) % 5 == 0: # Evaluate every 5 epochs
            model.eval()
            all_precisions_at_k = []
            all_recalls_at_k = []
            
            # Validation target is Week 9 (index 8)
            val_target_idx = MAX_WEEKS_TOTAL - 1 # Week 9 is at index 8
            val_history_start_idx = val_target_idx - HISTORICAL_WINDOW_SIZE

            if val_history_start_idx >= 0 and val_target_idx < len(all_weekly_snapshots):
                val_historical_window = all_weekly_snapshots[val_history_start_idx : val_target_idx]
                val_target_snapshot = all_weekly_snapshots[val_target_idx] # This is Week 9 data
                
                true_positives_in_val_target = {}
                if ('user', 'interacts_with', 'item') in val_target_snapshot.edge_types and \
                    val_target_snapshot['user', 'interacts_with', 'item'].edge_index.numel() > 0:
                    val_edges = val_target_snapshot['user', 'interacts_with', 'item'].edge_index
                    for edge_i in range(val_edges.size(1)):
                        u, it = val_edges[0, edge_i].item(), val_edges[1, edge_i].item()
                        true_positives_in_val_target.setdefault(u, set()).add(it)
                
                # Evaluate on all users who had interactions in the validation target week, or a sample
                users_to_eval_val = sorted(list(true_positives_in_val_target.keys()))
                if not users_to_eval_val and num_unique_users > 0: 
                    users_to_eval_val = list(range(min(10, num_unique_users))) # Fallback: eval a few users

                with torch.no_grad():
                    # --- Get temporal embeddings for validation ---
                    # This simulation is still needed as model.forward() is for training loss.
                    user_embeddings_val_list_hist = []
                    item_embeddings_val_list_hist = []
                    valid_history_for_eval = True
                    for snapshot_val_hist in val_historical_window:
                        if snapshot_val_hist is None: 
                            valid_history_for_eval = False; break
                        x_dict_val = snapshot_val_hist.x_dict
                        edge_index_dict_val = snapshot_val_hist.edge_index_dict
                        
                        raw_z_dict_month_val = model.gnn_encoder(x_dict_val, edge_index_dict_val)
                        
                        # Process user features (handle None from GNN)
                        user_feat_from_gnn = raw_z_dict_month_val.get('user')
                        if user_feat_from_gnn is None:
                            if 'user' not in x_dict_val or x_dict_val['user'] is None:
                                print(f"Epoch {epoch+1} Eval: GNN returned None for 'user' and no initial features in historical snapshot. Skipping this snapshot.")
                                valid_history_for_eval = False; break # Or handle differently, e.g. skip snapshot
                            initial_user_feat = x_dict_val['user']
                            user_feat_processed_gnn_stage = model.user_initial_proj(initial_user_feat)
                        else:
                            user_feat_processed_gnn_stage = user_feat_from_gnn
                        
                        processed_user_feat_val = model.user_relu(user_feat_processed_gnn_stage)
                        processed_user_feat_val = model.user_dropout(processed_user_feat_val)
                        processed_user_feat_val = model.user_bn(processed_user_feat_val)

                        # Process item features
                        item_feat_from_gnn_val = raw_z_dict_month_val['item'] # Assume item is always present
                        processed_item_feat_val = model.item_relu(item_feat_from_gnn_val)
                        processed_item_feat_val = model.item_dropout(processed_item_feat_val)
                        processed_item_feat_val = model.item_bn(processed_item_feat_val)
                        
                        user_embeddings_val_list_hist.append(processed_user_feat_val)
                        item_embeddings_val_list_hist.append(processed_item_feat_val)
                    
                    if not valid_history_for_eval or not user_embeddings_val_list_hist or not item_embeddings_val_list_hist:
                        print(f"Epoch {epoch+1} Eval: Could not generate embeddings for validation. Skipping.")
                        continue
                    
                    user_static_seq_val = torch.stack(user_embeddings_val_list_hist, dim=0)
                    user_src_val = model.pos_encoder(user_static_seq_val)
                    user_trg_val = user_embeddings_val_list_hist[-1].unsqueeze(0)
                    user_temporal_embeds_val = model.transformer(user_src_val, user_trg_val).squeeze(0)
                    all_item_embeds_val = item_embeddings_val_list_hist[-1]
                    # --- End temporal embedding generation for validation ---

                    for user_id in users_to_eval_val:
                        if user_id >= user_temporal_embeds_val.size(0):
                            continue # User might not have been in historical data if not all users are active every week
                            
                        user_emb = user_temporal_embeds_val[user_id]
                        _, top_k_indices, _ = model.predict_top_k_for_user(user_emb, all_item_embeds_val, k=TOP_K)
                        predicted_top_k_items = top_k_indices.tolist()
                        
                        true_items_for_user = true_positives_in_val_target.get(user_id, set())
                        p_at_k, r_at_k = calculate_precision_recall_at_k(predicted_top_k_items, true_items_for_user, k=TOP_K)
                        all_precisions_at_k.append(p_at_k)
                        all_recalls_at_k.append(r_at_k)
                
                if all_precisions_at_k: # Check if any users were evaluated
                    avg_p_at_k = np.mean(all_precisions_at_k)
                    avg_r_at_k = np.mean(all_recalls_at_k)
                    print(f"Epoch {epoch+1} Eval (Week {MAX_WEEKS_TOTAL}): Avg Precision@{TOP_K}: {avg_p_at_k:.4f}, Avg Recall@{TOP_K}: {avg_r_at_k:.4f}")
                else:
                    print(f"Epoch {epoch+1} Eval (Week {MAX_WEEKS_TOTAL}): No users evaluated or no predictions made.")
            else:
                print(f"Epoch {epoch+1}: Not enough data or history for Week {MAX_WEEKS_TOTAL} validation.")

    print("\nTraining finished.")
    print(f"Data was loaded from: {CSV_FILE_PATH}")
    print(f"Model trained to predict interaction counts (MSE loss) for weeks {1} to {TRAIN_UP_TO_WEEK}.")
    print(f"Top-K evaluation performed on week {MAX_WEEKS_TOTAL} using model.predict_top_k_for_user().")

if __name__ == '__main__':
    main() 