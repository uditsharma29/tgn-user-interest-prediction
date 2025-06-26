import torch
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

# --- Constants that might be shared or derived (can be passed as args too) ---
# These dimensions are for the randomly generated features if not learned by model
USER_FEAT_DIM_DEFAULT = 16 
ITEM_FEAT_DIM_DEFAULT = 16

def generate_monthly_snapshot(monthly_data):
    """
    Generate a HeteroData object as snapshot of one month.

    Args:
        monthly_data (list): List of pandas dataframes with trade
                             and features' data for one month.

    Returns:
        HeteroData: Object containing node features and edge attributes.
    """
    monthly_snp = HeteroData()

    # Ingesting the data
    trade_figs = monthly_data[0]
    exporters_features_df = monthly_data[1]
    importers_features_df = monthly_data[2]
    edge_attrs_df = monthly_data[3]

    # Creating the nodes for exporters
    # Ensure exp_ids are sorted to match feature order if features are pre-sorted by id
    unique_exp_ids = sorted(trade_figs['exp_id'].unique())
    exp_id_map = {id_val: i for i, id_val in enumerate(unique_exp_ids)}
    
    # Assuming exporters_features_df is indexed by original exporter IDs or has a column for it
    # For simplicity, let's assume it's ordered by unique_exp_ids or has an 'id' column to map
    # If exporters_features_df contains features for all potential exporters, filter or map them
    # Here, we'll assume exporters_features_df is already aligned with unique_exp_ids
    
    # Create a mapping from original exporter IDs to new contiguous 0-indexed IDs
    exp_ids_tensor = torch.arange(len(unique_exp_ids), dtype=torch.int64) # contiguous IDs

    # Check if exporters_features_df is indexed by 'exp_id' or has it as a column
    if 'exp_id' in exporters_features_df.columns:
        exporters_features_df = exporters_features_df.set_index('exp_id').loc[unique_exp_ids].reset_index()
        exporters_ftrs_arr = exporters_features_df.drop(columns=['exp_id']).values
    elif exporters_features_df.index.name == 'exp_id' or exporters_features_df.index.name is None : #assuming index is exp_id if not named
        # Ensure the order matches unique_exp_ids
        exporters_features_df = exporters_features_df.reindex(unique_exp_ids)
        exporters_ftrs_arr = exporters_features_df.values
    else:
        # Fallback: assuming the order is correct and shape matches
        exporters_ftrs_arr = exporters_features_df.values

    if exporters_ftrs_arr.shape[0] != len(unique_exp_ids):
        raise ValueError(f"Mismatch in number of exporter IDs ({len(unique_exp_ids)}) and exporter features ({exporters_ftrs_arr.shape[0]})")

    exporters_ftrs_tensor = torch.tensor(exporters_ftrs_arr.astype(np.float32), dtype=torch.float)
    monthly_snp['exp_id'].x = exporters_ftrs_tensor
    monthly_snp['exp_id'].num_nodes = len(unique_exp_ids)


    # Creating the nodes for importers
    unique_imp_ids = sorted(trade_figs['imp_id'].unique())
    imp_id_map = {id_val: i for i, id_val in enumerate(unique_imp_ids)}

    imp_ids_tensor = torch.arange(len(unique_imp_ids), dtype=torch.int64)

    if 'imp_id' in importers_features_df.columns:
        importers_features_df = importers_features_df.set_index('imp_id').loc[unique_imp_ids].reset_index()
        importers_ftrs_arr = importers_features_df.drop(columns=['imp_id']).values
    elif importers_features_df.index.name == 'imp_id' or importers_features_df.index.name is None:
        importers_features_df = importers_features_df.reindex(unique_imp_ids)
        importers_ftrs_arr = importers_features_df.values
    else:
        importers_ftrs_arr = importers_features_df.values
        
    if importers_ftrs_arr.shape[0] != len(unique_imp_ids):
        raise ValueError(f"Mismatch in number of importer IDs ({len(unique_imp_ids)}) and importer features ({importers_ftrs_arr.shape[0]})")

    importers_ftrs_tensor = torch.tensor(importers_ftrs_arr.astype(np.float32), dtype=torch.float)
    monthly_snp['imp_id'].x = importers_ftrs_tensor
    monthly_snp['imp_id'].num_nodes = len(unique_imp_ids)

    # Map original trade_figs exp_id and imp_id to new 0-indexed IDs
    mapped_exp_ids = torch.tensor([exp_id_map[id_val] for id_val in trade_figs['exp_id'].values], dtype=torch.long)
    mapped_imp_ids = torch.tensor([imp_id_map[id_val] for id_val in trade_figs['imp_id'].values], dtype=torch.long)

    # Creating the edges
    edge_index = torch.stack([
        mapped_exp_ids,
        mapped_imp_ids
    ], dim=0)

    monthly_snp['exp_id', 'volume', 'imp_id'].edge_index = edge_index

    vol = torch.from_numpy(trade_figs['volume'].values.astype(np.float32)).to(torch.float)
    monthly_snp['exp_id', 'volume', 'imp_id'].edge_label = vol

    # Assuming edge_attrs_df corresponds to the order in trade_figs
    edge_attrs_arr = edge_attrs_df.values.astype(np.float32)
    edge_attrs_tensor = torch.tensor(edge_attrs_arr).to(torch.float)
    monthly_snp['exp_id', 'volume', 'imp_id'].edge_attr = edge_attrs_tensor # Corrected from 'edge_attrs'

    monthly_snp['exp_id', 'volume', 'imp_id'].edge_label_index = monthly_snp['exp_id', 'volume', 'imp_id'].edge_index.clone()

    monthly_snp = ToUndirected()(monthly_snp)
    # The article mentions deleting 'rev_volume' edge_label.
    # This implies that ToUndirected() creates reverse edges with labels.
    # We need to check the exact key ToUndirected creates for reverse labels.
    # Typically, it might be ('imp_id', 'rev_volume', 'exp_id').edge_label
    if ('imp_id', 'rev_volume', 'exp_id') in monthly_snp.edge_types and 'edge_label' in monthly_snp[('imp_id', 'rev_volume', 'exp_id')]:
         del monthly_snp[('imp_id', 'rev_volume', 'exp_id')]['edge_label']
    elif ('imp_id', 'rev_volume', 'exp_id') in monthly_snp.edge_types and 'edge_label' in monthly_snp[('imp_id', 'rev_volume', 'exp_id')]:
        del monthly_snp[('imp_id', 'rev_volume', 'exp_id')]['edge_label']


    return monthly_snp

def create_dummy_data_for_month(num_countries=10, num_trades=20, exporter_feat_dim=5, importer_feat_dim=5, edge_feat_dim=3):
    """
    Generates dummy data for one month.
    Returns a list of 4 pandas DataFrames:
    1. trade_figs: ['exp_id', 'imp_id', 'volume']
    2. exporters_features: DataFrame with exporter features (index=exp_id)
    3. importers_features: DataFrame with importer features (index=imp_id)
    4. edge_attrs: DataFrame with edge features (aligns with trade_figs)
    """
    all_country_ids = np.arange(num_countries)

    # Trade figures
    exp_ids_trade = np.random.choice(all_country_ids, num_trades)
    imp_ids_trade = np.random.choice(all_country_ids, num_trades)
    # Ensure exporter and importer are not the same for a trade
    for i in range(num_trades):
        while exp_ids_trade[i] == imp_ids_trade[i]:
            imp_ids_trade[i] = np.random.choice(all_country_ids)
            
    trade_figs_df = pd.DataFrame({
        'exp_id': exp_ids_trade,
        'imp_id': imp_ids_trade,
        'volume': np.random.rand(num_trades) * 1000
    })

    # Unique exporter and importer IDs from the actual trades this month
    unique_exp_ids_trade = sorted(trade_figs_df['exp_id'].unique())
    unique_imp_ids_trade = sorted(trade_figs_df['imp_id'].unique())

    # Exporter features
    # Features for all potential countries that *could* be exporters
    exporters_features_df = pd.DataFrame(
        np.random.rand(len(all_country_ids), exporter_feat_dim),
        index=all_country_ids, # Use all_country_ids as index
        columns=[f'exp_feat_{j}' for j in range(exporter_feat_dim)]
    )
    exporters_features_df.index.name = 'exp_id'
    # We will ensure generate_monthly_snapshot selects features for unique_exp_ids_trade

    # Importer features
    # Features for all potential countries that *could* be importers
    importers_features_df = pd.DataFrame(
        np.random.rand(len(all_country_ids), importer_feat_dim),
        index=all_country_ids, # Use all_country_ids as index
        columns=[f'imp_feat_{j}' for j in range(importer_feat_dim)]
    )
    importers_features_df.index.name = 'imp_id'
    # We will ensure generate_monthly_snapshot selects features for unique_imp_ids_trade


    # Edge attributes (must match the number of trades)
    edge_attrs_df = pd.DataFrame(
        np.random.rand(num_trades, edge_feat_dim),
        columns=[f'edge_feat_{j}' for j in range(edge_feat_dim)]
    )

    return [trade_figs_df, exporters_features_df, importers_features_df, edge_attrs_df]

def load_interactions_from_csv(csv_path, static_user_feature_dim=32, static_event_feature_dim=64):
    """
    Loads interaction data from a CSV file.
    Creates mappings for ip_address to user_id and pixel_event_id to event_id (integer).
    Generates random static features for users and events.
    
    Args:
        csv_path (str): Path to the CSV file.
        static_user_feature_dim (int): Dimension for randomly generated static user features.
        static_event_feature_dim (int): Dimension for randomly generated static event features.
        
    Returns:
        tuple: (interactions_df, user_mapping, event_mapping, 
                user_feature_info, event_feature_info)
        interactions_df: DataFrame with original and mapped IDs.
        user_mapping: dict mapping ip_address to integer user_id.
        event_mapping: dict mapping original pixel_event_id (string) to integer event_id.
        user_feature_info (dict): {'num_unique': int, 'static_features': torch.Tensor}
        event_feature_info (dict): {'num_unique': int, 'static_features': torch.Tensor}
    """
    print(f"Loading interaction data from {csv_path}...")
    try:
        interactions_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: Data file {csv_path} not found.")
        raise
    
    if interactions_df.empty:
        print("Warning: The CSV file is empty.")
        user_feature_info = {'num_unique': 0, 'static_features': torch.empty(0, static_user_feature_dim)}
        event_feature_info = {'num_unique': 0, 'static_features': torch.empty(0, static_event_feature_dim)}
        return pd.DataFrame(), {}, {}, user_feature_info, event_feature_info

    unique_ips = interactions_df['ip_address'].unique()
    user_mapping = {ip: i for i, ip in enumerate(unique_ips)}
    interactions_df['user_id'] = interactions_df['ip_address'].map(user_mapping)
    num_unique_users = len(unique_ips)
    print(f"Found {num_unique_users} unique users (IPs).")

    static_user_features = torch.randn(num_unique_users, static_user_feature_dim)
    user_feature_info = {'num_unique': num_unique_users, 'static_features': static_user_features}

    if 'pixel_event_id' not in interactions_df.columns:
        raise ValueError("CSV file must contain 'pixel_event_id' column.")
        
    unique_event_ids_csv = interactions_df['pixel_event_id'].unique() 
    event_mapping = {pid: i for i, pid in enumerate(unique_event_ids_csv)}
    interactions_df['event_id'] = interactions_df['pixel_event_id'].map(event_mapping) 
    num_unique_events = len(unique_event_ids_csv)
    print(f"Found {num_unique_events} unique Event IDs (from pixel_event_id).")

    static_event_features = torch.randn(num_unique_events, static_event_feature_dim)
    event_feature_info = {'num_unique': num_unique_events, 'static_features': static_event_features}
    
    interactions_df = interactions_df.sort_values(by=['week_number', 'user_id']).reset_index(drop=True)
    
    print("Finished loading and mapping data.")
    return interactions_df, user_mapping, event_mapping, user_feature_info, event_feature_info

def create_weekly_heterodata_from_df(week_df, user_features_static, event_features_static):
    """
    Creates a HeteroData object for a specific week from the interaction DataFrame.
    This version creates both forward and reverse edges for compatibility with sampling.

    Args:
        week_df (pd.DataFrame): DataFrame of interactions for a single week.
        user_features_static (torch.Tensor): Static features for all users.
        event_features_static (torch.Tensor): Static features for all events.
    
    Returns:
        HeteroData: A graph object for the week with undirected edges.
    """
    snapshot = HeteroData()

    # Assign node features and number of nodes
    snapshot['user'].x = user_features_static
    snapshot['event'].x = event_features_static
    snapshot['user'].num_nodes = user_features_static.size(0)
    snapshot['event'].num_nodes = event_features_static.size(0)
    
    # Create forward edges
    user_indices = torch.tensor(week_df['user_id'].values, dtype=torch.long)
    event_indices = torch.tensor(week_df['event_id'].values, dtype=torch.long)
    edge_index_user_to_event = torch.stack([user_indices, event_indices], dim=0)
    
    snapshot['user', 'interacts_with', 'event'].edge_index = edge_index_user_to_event
    
    # Create reverse edges for undirected graph representation
    snapshot['event', 'rev_interacts_with', 'user'].edge_index = torch.stack([event_indices, user_indices], dim=0)

    return snapshot

# Example usage (can be run standalone to test CSV loading and one snapshot generation)
if __name__ == '__main__':
    # --- First, ensure 'user_event_interactions.csv' exists --- 
    # --- You might need to run generate_dataset_csv.py first ---
    csv_file = 'user_event_interactions.csv'
    try:
        interactions_df, user_map, item_map, n_users, n_items = load_interactions_from_csv(csv_file)
        print(f"\nLoaded data for {n_users} users and {n_items} items.")
        print("Sample of loaded DataFrame with mapped IDs:")
        print(interactions_df.head())

        # Generate static features (randomly for this example)
        static_user_features = torch.randn(n_users, USER_FEAT_DIM_DEFAULT)
        static_item_features = torch.randn(n_items, ITEM_FEAT_DIM_DEFAULT)

        # Create a snapshot for a specific week (e.g., week 1)
        week_1_df = interactions_df[interactions_df['week_number'] == 1]
        if not week_1_df.empty:
            hetero_snapshot_week_1 = create_weekly_heterodata_from_df(
                week_1_df, 
                static_user_features,
                static_item_features
            )
            print("\nGenerated HeteroData snapshot for Week 1:")
            print(hetero_snapshot_week_1)
            print(f"User nodes: {hetero_snapshot_week_1['user'].num_nodes}, Item nodes: {hetero_snapshot_week_1['item'].num_nodes}")
            print(f"Number of interactions in week 1 snapshot: {hetero_snapshot_week_1['user', 'interacts_with', 'item'].edge_index.size(1)}")
        else:
            print("\nNo data for Week 1 in the CSV to create a snapshot.")
            
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found. Please run generate_dataset_csv.py first.") 