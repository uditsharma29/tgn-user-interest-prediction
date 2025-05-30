import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import math

class GNNEncoder(nn.Module):
    def __init__(self, user_feature_dim, event_feature_dim, gnn_hidden_dim, num_gnn_layers, sage_aggr, sage_dropout_rate):
        super(GNNEncoder, self).__init__()
        self.num_gnn_layers = num_gnn_layers
        self.sage_dropout_rate = sage_dropout_rate

        self.user_convs = nn.ModuleList()
        self.event_convs = nn.ModuleList()

        self.user_initial_proj = nn.Linear(user_feature_dim, gnn_hidden_dim) if user_feature_dim != gnn_hidden_dim else None
        self.event_initial_proj = nn.Linear(event_feature_dim, gnn_hidden_dim) if event_feature_dim != gnn_hidden_dim else None

        for _ in range(num_gnn_layers):
            self.user_convs.append(SAGEConv((-1, -1), gnn_hidden_dim, aggr=sage_aggr))
            self.event_convs.append(SAGEConv((-1, -1), gnn_hidden_dim, aggr=sage_aggr))

        self.user_bns = nn.ModuleList([nn.BatchNorm1d(gnn_hidden_dim) for _ in range(num_gnn_layers)])
        self.event_bns = nn.ModuleList([nn.BatchNorm1d(gnn_hidden_dim) for _ in range(num_gnn_layers)])

    def forward(self, x_user, x_event, edge_index):
        if self.user_initial_proj:
            x_user = self.user_initial_proj(x_user)
        if self.event_initial_proj:
            x_event = self.event_initial_proj(x_event)

        for i in range(self.num_gnn_layers):
            # Users are target, events are source. SAGEConv expects (x_src, x_dst), so (x_event, x_user)
            # Edge index needs to be flipped for this: (event_idx, user_idx)
            x_user_updated = self.user_convs[i]((x_event, x_user), edge_index[[1,0]])
            x_user = self.user_bns[i](x_user_updated)
            x_user = F.relu(x_user)
            x_user = F.dropout(x_user, p=self.sage_dropout_rate, training=self.training)

            # Events are target, users are source
            x_event_updated = self.event_convs[i]((x_user, x_event), edge_index)
            x_event = self.event_bns[i](x_event_updated)
            x_event = F.relu(x_event)
            x_event = F.dropout(x_event, p=self.sage_dropout_rate, training=self.training)
            
        return x_user, x_event

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size (num_users), features (d_model)]
        x = x + self.pe[:x.size(0), :]
        return x

class UserInteractionPredictor(nn.Module):
    def __init__(self, user_feature_dim, event_feature_dim,
                 gnn_hidden_dim, num_gnn_layers, sage_aggr, sage_dropout_rate,
                 transformer_d_model, transformer_nhead, transformer_num_layers, transformer_dropout,
                 num_unique_users, num_global_events):
        super(UserInteractionPredictor, self).__init__()

        self.gnn_encoder = GNNEncoder(user_feature_dim, event_feature_dim, gnn_hidden_dim,
                                      num_gnn_layers, sage_aggr, sage_dropout_rate)
        
        self.num_global_events = num_global_events
        self.transformer_d_model = transformer_d_model
        self.gnn_hidden_dim = gnn_hidden_dim

        # Always use transformer components
        self.gnn_to_transformer_proj = nn.Linear(gnn_hidden_dim, transformer_d_model) if gnn_hidden_dim != transformer_d_model else None
        self.pos_encoder = PositionalEncoding(transformer_d_model)
        encoder_layer_args = {
            'd_model': transformer_d_model,
            'nhead': transformer_nhead,
            'dim_feedforward': transformer_d_model * 4,
            'dropout': transformer_dropout,
            'batch_first': False
        }
        try:
            encoder_layers = nn.TransformerEncoderLayer(**encoder_layer_args, activation=F.gelu)
        except TypeError:
            encoder_layers = nn.TransformerEncoderLayer(**encoder_layer_args)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_num_layers)
        
        # Modified prediction head to use similarity
        self.user_to_event_scores_mlp = nn.Linear(transformer_d_model + 1, 1)  # +1 for similarity score

    def _get_embeddings_for_prediction(self, input_snapshots_for_gnn):
        # Helper to run GNN and Transformer
        all_user_gnn_embs_list = []
        all_event_gnn_embs_list = []
        
        for snapshot_data in input_snapshots_for_gnn:
            user_x = snapshot_data['user'].x
            event_x = snapshot_data['event'].x
            edge_idx = snapshot_data['user', 'interacts_with', 'event'].edge_index
            current_user_gnn_emb, current_event_gnn_emb = self.gnn_encoder(user_x, event_x, edge_idx)
            all_user_gnn_embs_list.append(current_user_gnn_emb)
            all_event_gnn_embs_list.append(current_event_gnn_emb)
        
        if len(all_user_gnn_embs_list) < 2:
            raise ValueError("Need at least 2 snapshots for transformer (history + target)")

        # Process historical snapshots through transformer
        user_embeddings_seq_hist = torch.stack(all_user_gnn_embs_list[:-1], dim=0)
        if self.gnn_to_transformer_proj:
            user_embeddings_seq_hist = self.gnn_to_transformer_proj(user_embeddings_seq_hist)
        if self.pos_encoder:
            user_embeddings_seq_hist = self.pos_encoder(user_embeddings_seq_hist)
        
        transformed_user_output = self.transformer_encoder(user_embeddings_seq_hist)
        user_embeddings_for_prediction = transformed_user_output[-1, :, :]
            
        # Use the last event embeddings
        event_embeddings_for_prediction = all_event_gnn_embs_list[-1]
        
        return user_embeddings_for_prediction, event_embeddings_for_prediction

    def get_temporal_embeddings(self, input_snapshots):
        """
        Get temporal embeddings for both users and events.
        Requires at least 2 snapshots (history + target).
        """
        if not input_snapshots:
            raise ValueError("input_snapshots cannot be empty.")

        if len(input_snapshots) < 2:
            raise ValueError("Need at least 2 snapshots (history + target) for transformer")
        
        return self._get_embeddings_for_prediction(input_snapshots)

    def _compute_similarity_scores(self, user_embeddings, event_embeddings, user_indices, event_indices):
        # Get embeddings for the specified user-event pairs
        user_embs = user_embeddings[user_indices]
        event_embs = event_embeddings[event_indices]
        
        # Compute cosine similarity
        similarity = F.cosine_similarity(user_embs, event_embs, dim=1)
        
        # Combine user embeddings with similarity scores
        combined = torch.cat([user_embs, similarity.unsqueeze(1)], dim=1)
        
        # Get final scores
        scores = self.user_to_event_scores_mlp(combined).squeeze()
        
        return scores

    def forward(self, model_input_snapshots, positive_user_event_indices, negative_event_indices):
        """
        Forward pass for training.
        
        Args:
            model_input_snapshots: List of HeteroData objects for historical weeks + target week
            positive_user_event_indices: Dict with keys 'user' and 'event' containing indices of positive interactions
            negative_event_indices: Dict with keys 'user' and 'event' containing indices of negative samples
            
        Returns:
            Dict containing:
                - positive_scores: Tensor of scores for positive user-event pairs
                - negative_scores: Tensor of scores for negative user-event pairs
        """
        # Get user and event embeddings for prediction
        user_embeddings, event_embeddings = self._get_embeddings_for_prediction(model_input_snapshots)
        
        # Get scores for positive and negative pairs
        positive_scores = self._compute_similarity_scores(
            user_embeddings, 
            event_embeddings,
            positive_user_event_indices['user'],
            positive_user_event_indices['event']
        )
        
        negative_scores = self._compute_similarity_scores(
            user_embeddings,
            event_embeddings,
            negative_event_indices['user'],
            negative_event_indices['event']
        )
        
        return {
            'positive_scores': positive_scores,
            'negative_scores': negative_scores
        }

    def predict_top_k_for_user(self, single_user_temporal_embedding, event_embeddings, k=5):
        self.eval()
        with torch.no_grad():
            # Compute similarities for all events
            similarities = F.cosine_similarity(
                single_user_temporal_embedding.unsqueeze(0).expand(len(event_embeddings), -1),
                event_embeddings,
                dim=1
            )
            
            # Combine user embedding with similarities
            combined = torch.cat([
                single_user_temporal_embedding.unsqueeze(0).expand(len(event_embeddings), -1),
                similarities.unsqueeze(1)
            ], dim=1)
            
            # Get scores for all events
            event_scores = self.user_to_event_scores_mlp(combined).squeeze()
            
            # Get top-k events
            actual_k = min(k, self.num_global_events)
            if actual_k <= 0:
                return [], []
            
            top_k_scores, top_k_indices = torch.topk(event_scores, actual_k)
            return top_k_indices.tolist(), top_k_scores.tolist()