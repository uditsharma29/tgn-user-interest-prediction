import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import math

class GNNEncoder(nn.Module):
    """
    Graph Neural Network Encoder using SAGEConv to generate static node embeddings.
    This module is intended to be wrapped by `to_hetero` to handle heterogeneous graphs.
    Its forward pass should ONLY contain the core GNN convolution.
    Other operations like ReLU, Dropout, BatchNorm should be applied outside,
    after `to_hetero` has processed the output.
    """
    def __init__(self, in_channels, hidden_channels, sage_aggr='mean'): # Removed unused params
        super().__init__()
        self.conv1 = SAGEConv((-1,-1), hidden_channels, aggr=sage_aggr)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass for a single homogeneous graph segment using SAGEConv.
        Args:
            x (Tensor): Node features.
            edge_index (Tensor): Edge connectivity.
            edge_attr (Tensor, optional): Edge features (ignored by SAGEConv).
        Returns:
            Tensor: Raw node embeddings after SAGEConv.
        """
        x_out = self.conv1(x, edge_index)
        return x_out

class PositionalEncoding(nn.Module):
    """
    Standard Positional Encoding for Transformer.
    Adds positional information to a sequence of embeddings.
    """
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape [seq_len, batch_size (num_nodes), embedding_dim].
        Returns:
            Tensor: Output tensor with added positional encodings.
        """
        x = x + self.pe[:x.size(0), :] 
        return self.dropout(x)

class UserInteractionPredictor(nn.Module):
    """
    Model for predicting user-item interactions, aiming for Top-K recommendations.
    Uses a SAGE-based GNN Encoder and a Transformer.
    """
    def __init__(self, metadata,
                 user_feat_dim, item_feat_dim, 
                 gnn_hidden_channels, 
                 transformer_d_model, 
                 tr_heads, tr_encoder_layers, tr_decoder_layers, 
                 num_global_items, 
                 sage_aggr='mean', sage_dropout_rate=0.1, gnn_momentum_bn=0.1,
                 tr_dropout=0.1, 
                 transformer_max_seq_len=50):
        super().__init__()

        self.gnn_encoder_module = GNNEncoder(
            in_channels=-1, 
            hidden_channels=gnn_hidden_channels, 
            sage_aggr=sage_aggr
        )
        self.gnn_encoder = to_hetero(self.gnn_encoder_module, metadata=metadata, aggr='sum')

        self.user_feat_dim = user_feat_dim
        self.item_feat_dim = item_feat_dim
        self.gnn_hidden_channels = gnn_hidden_channels # Store for projection logic

        if self.user_feat_dim != self.gnn_hidden_channels:
            self.user_initial_proj = nn.Linear(self.user_feat_dim, self.gnn_hidden_channels)
        else:
            self.user_initial_proj = nn.Identity()

        self.user_relu = nn.ReLU()
        self.user_dropout = nn.Dropout(p=sage_dropout_rate)
        self.user_bn = nn.BatchNorm1d(self.gnn_hidden_channels, momentum=gnn_momentum_bn)
        
        self.item_relu = nn.ReLU()
        self.item_dropout = nn.Dropout(p=sage_dropout_rate)
        self.item_bn = nn.BatchNorm1d(self.gnn_hidden_channels, momentum=gnn_momentum_bn)

        if transformer_d_model != self.gnn_hidden_channels:
            print(f"Warning: UserInteractionPredictor transformer_d_model ({transformer_d_model}) does not match gnn_hidden_channels ({self.gnn_hidden_channels}). Align or use projection.")
        self.transformer_d_model = transformer_d_model

        self.pos_encoder = PositionalEncoding(self.transformer_d_model, dropout=tr_dropout, max_len=transformer_max_seq_len)

        self.transformer = nn.Transformer(
            d_model=self.transformer_d_model, 
            nhead=tr_heads,
            num_encoder_layers=tr_encoder_layers, 
            num_decoder_layers=tr_decoder_layers,
            dropout=tr_dropout, 
            batch_first=False
        )
        
        self.num_global_items = num_global_items
        self.user_to_item_scores_mlp = nn.Linear(self.transformer_d_model, self.num_global_items)

    def forward(self, historical_snapshots, target_snapshot_metadata_provider):
        user_embeds_static_list = []
        item_embeds_static_list = []

        num_user_nodes_target = target_snapshot_metadata_provider['user'].num_nodes
        num_item_nodes_target = target_snapshot_metadata_provider['item'].num_nodes

        for snapshot in historical_snapshots:
            x_dict = snapshot.x_dict
            edge_index_dict = snapshot.edge_index_dict
            
            raw_z_dict_month = self.gnn_encoder(x_dict, edge_index_dict)
            
            # Handle user features
            user_embeddings_from_gnn = raw_z_dict_month.get('user')
            if user_embeddings_from_gnn is None:
                if 'user' not in x_dict or x_dict['user'] is None:
                    raise ValueError("GNN returned None for 'user' and no initial 'user' features in x_dict.")
                initial_user_feat = x_dict['user']
                user_feat_processed_gnn_stage = self.user_initial_proj(initial_user_feat)
            else:
                user_feat_processed_gnn_stage = user_embeddings_from_gnn
            
            user_feat = self.user_relu(user_feat_processed_gnn_stage)
            user_feat = self.user_dropout(user_feat)
            user_feat = self.user_bn(user_feat)
            
            # Handle item features (assuming GNN always outputs for 'item' as it's a dst_node_type)
            item_feat_from_gnn = raw_z_dict_month['item']
            item_feat = self.item_relu(item_feat_from_gnn)
            item_feat = self.item_dropout(item_feat)
            item_feat = self.item_bn(item_feat)

            processed_z_dict_month = {'user': user_feat, 'item': item_feat}
            
            if processed_z_dict_month['user'].size(0) != num_user_nodes_target:
                raise ValueError(f"User node count mismatch. Expected {num_user_nodes_target}, got {processed_z_dict_month['user'].size(0)}")
            if processed_z_dict_month['item'].size(0) != num_item_nodes_target:
                 raise ValueError(f"Item node count mismatch. Expected {num_item_nodes_target}, got {processed_z_dict_month['item'].size(0)}")

            user_embeds_static_list.append(processed_z_dict_month['user'])
            item_embeds_static_list.append(processed_z_dict_month['item'])

        if not user_embeds_static_list or not item_embeds_static_list:
            return torch.empty((0,), device=self.user_to_item_scores_mlp.weight.device if hasattr(self.user_to_item_scores_mlp, 'weight') else 'cpu')

        user_static_seq = torch.stack(user_embeds_static_list, dim=0)
        item_static_seq = torch.stack(item_embeds_static_list, dim=0)

        user_src_for_transformer = self.pos_encoder(user_static_seq)
        
        user_trg_for_decoder = user_embeds_static_list[-1].unsqueeze(0)
        user_temporal_embeds_seq = self.transformer(user_src_for_transformer, user_trg_for_decoder)
        user_temporal_embeds_flat = user_temporal_embeds_seq.squeeze(0)
        item_embeds_for_prediction_time = item_embeds_static_list[-1]

        if ('user', 'interacts_with', 'item') in target_snapshot_metadata_provider.edge_types and \
            target_snapshot_metadata_provider['user', 'interacts_with', 'item'].edge_label_index.numel() > 0:
            
            edge_label_index = target_snapshot_metadata_provider['user', 'interacts_with', 'item'].edge_label_index
            src_user_indices = edge_label_index[0]
            tgt_item_indices = edge_label_index[1]

            user_embeds_for_observed = user_temporal_embeds_flat[src_user_indices]
            item_embeds_for_observed = item_embeds_for_prediction_time[tgt_item_indices]
            
            scores_for_observed_edges = (user_embeds_for_observed * item_embeds_for_observed).sum(dim=-1)
            return scores_for_observed_edges
        else:
            return torch.empty((0,), device=user_temporal_embeds_flat.device)
        
    def predict_top_k_for_user(self, user_embedding, all_item_embeddings, k=5):
        scores = self.user_to_item_scores_mlp(user_embedding)
        
        if scores.numel() == 0 :
             return torch.empty((0,)), torch.empty((0,),dtype=torch.long), torch.empty((0,))

        k_effective = min(k, scores.size(0))
        if k_effective <= 0:
            return torch.empty((scores.size(0),)), torch.empty((0,),dtype=torch.long), torch.empty((0,))

        top_k_scores, top_k_indices = torch.topk(scores, k=k_effective)
        return scores, top_k_indices, top_k_scores 