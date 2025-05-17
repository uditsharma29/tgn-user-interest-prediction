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

def load_interactions_from_csv(csv_path):
    """
    Loads interaction data from a CSV file.
    Creates mappings for ip_address to user_id and item_id to item_id (integer).
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        tuple: (interactions_df, user_mapping, item_mapping, 
                num_unique_users, num_unique_items)
        interactions_df: DataFrame with original and mapped IDs.
        user_mapping: dict mapping ip_address to integer user_id.
        item_mapping: dict mapping original item_id (string) to integer item_id.
    """
    print(f"Loading interaction data from {csv_path}...")
    interactions_df = pd.read_csv(csv_path)
    
    # Create user mapping (ip_address -> user_id)
    unique_ips = interactions_df['ip_address'].unique()
    user_mapping = {ip: i for i, ip in enumerate(unique_ips)}
    interactions_df['user_id'] = interactions_df['ip_address'].map(user_mapping)
    num_unique_users = len(unique_ips)
    print(f"Found {num_unique_users} unique users (IPs).")

    # Create item mapping (original item_id -> integer item_id)
    unique_item_ids_csv = interactions_df['item_id'].unique() # Was pixel_event_id
    item_mapping = {pid: i for i, pid in enumerate(unique_item_ids_csv)} # Was event_mapping
    # The new integer mapped column will also be 'item_id', replacing original if it was string.
    # If original 'item_id' from CSV could be integer, ensure this mapping is what's intended.
    # For clarity, let's use a distinct mapped column name if needed, e.g., 'mapped_item_id',
    # but current project uses same name for the mapped ID column (e.g. user_id).
    interactions_df['item_id_mapped'] = interactions_df['item_id'].map(item_mapping) # Was event_id, using item_id from CSV
    num_unique_items = len(unique_item_ids_csv) # Was num_unique_events
    print(f"Found {num_unique_items} unique Item IDs.") # Was pixel event IDs
    
    # Sort by week and user for potentially easier processing later (optional)
    interactions_df = interactions_df.sort_values(by=['week_number', 'user_id']).reset_index(drop=True)
    
    print("Finished loading and mapping data.")
    return interactions_df, user_mapping, item_mapping, num_unique_users, num_unique_items

def create_weekly_heterodata_from_df(week_df, num_total_users, num_total_items, 
                                     user_features_static, item_features_static):
    """
    Creates a HeteroData object for a specific week from the interaction DataFrame.
    Assumes week_df is a DataFrame filtered for a single week and contains mapped 'user_id' and 'item_id_mapped'.
    User and Item features are provided externally (e.g., generated once).

    Args:
        week_df (pd.DataFrame): DataFrame of interactions for a single week.
        num_total_users (int): Total number of unique users across all data.
        num_total_items (int): Total number of unique items across all data. (Was num_total_events)
        user_features_static (torch.Tensor): Static features for all users.
        item_features_static (torch.Tensor): Static features for all items. (Was event_features_static)
        
    Returns:
        HeteroData: Graph object for the week.
    """
    snapshot = HeteroData()

    # Assign static node features
    snapshot['user'].x = user_features_static
    snapshot['user'].num_nodes = num_total_users
    
    snapshot['item'].x = item_features_static # Was 'event'
    snapshot['item'].num_nodes = num_total_items # Was 'event', num_total_events

    if not week_df.empty:
        # Edges (user -> item interactions)
        # Columns 'user_id' and 'item_id_mapped' must exist and be mapped integer IDs
        edge_index_user_item = torch.stack([
            torch.tensor(week_df['user_id'].values, dtype=torch.long),
            torch.tensor(week_df['item_id_mapped'].values, dtype=torch.long) # Was 'event_id'
        ], dim=0)

        # Edge label (e.g., number of interactions)
        # Ensure the column name for interaction count is correct (e.g., 'n_interactions')
        edge_label = torch.tensor(week_df['n_interactions'].values, dtype=torch.float32) # Was 'n_events'

        snapshot['user', 'interacts_with', 'item'].edge_index = edge_index_user_item # Was ('user', 'interacts_with', 'event')
        snapshot['user', 'interacts_with', 'item'].edge_label = edge_label # Was ('user', 'interacts_with', 'event')
        
        # Store edge_label_index for supervised learning, typically same as edge_index for link-level tasks
        snapshot['user', 'interacts_with', 'item'].edge_label_index = edge_index_user_item.clone()
    else:
        # Ensure all expected keys are present even if no interactions
        snapshot['user', 'interacts_with', 'item'].edge_index = torch.empty((2,0), dtype=torch.long)
        snapshot['user', 'interacts_with', 'item'].edge_label = torch.empty((0,), dtype=torch.float32)
        snapshot['user', 'interacts_with', 'item'].edge_label_index = torch.empty((2,0), dtype=torch.long)

    # Optional: Convert to undirected if necessary, though for user-item it might be kept directed.
    # If converting, be mindful of how edge attributes/labels are handled for reverse edges.
    # snapshot = ToUndirected()(snapshot) 

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
                n_users, 
                n_items, 
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