# Temporal Heterogeneous Graph Model for User Interest Prediction

This project implements a temporal graph neural network (TGN) model to predict the top 5 item IDs a user might be interested in during the next week. It uses PyTorch and PyTorch Geometric, employing a neighborhood sampling strategy for scalable GNN training.

## Project Goal

The primary goal is to model user-item interactions over time. The model first learns general user and item embeddings from the entire interaction history using a scalable GNN. It then uses a Transformer to capture temporal patterns from sequences of these learned embeddings to predict future user interests.

## File Structure

```
.
├── generate_dataset_csv.py  # Generates a dummy CSV dataset of user-item interactions.
├── data_utils.py            # Utilities for loading and processing data into PyG HeteroData snapshots.
├── model.py                 # Defines the GNN encoder, Transformer, and the main UserInteractionPredictor model.
├── train.py                 # Main script for training the model and evaluating Top-K predictions.
├── requirements.txt         # Python package dependencies.
└── user_event_interactions.csv # (Generated by generate_dataset_csv.py) The input dataset.
```

## Setup and Installation

1.  **Create a Python virtual environment** (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install dependencies**:
    This project requires PyTorch and several PyTorch Geometric libraries. The installation is sensitive to the OS and hardware (CPU/CUDA/Apple Silicon).

    *   **Step 1: Install PyTorch**: Follow the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) to install the correct version for your system.

    *   **Step 2: Install PyG Libraries**: After installing PyTorch, install the required libraries. The `NeighborSampler` used in this project requires either `pyg-lib` or `torch-sparse`.
        ```bash
        pip install -r requirements.txt
        ```
        Then, install the necessary backend library. Find your installed PyTorch version (e.g., 2.3.0) and platform (`cpu`, `cu121`, etc.) and run the appropriate command from the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). For example, for PyTorch 2.3.0 and CPU:
        ```bash
        pip install torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
        ```
        **Note for Apple Silicon (M1/M2/M3) users**: Pre-built wheels may not be available. If the command above fails, you may need to install the libraries from source as described in the PyG documentation.

## Running the Project

1.  **Generate the Dataset**:
    First, run the script to generate the dummy interaction data:
    ```bash
    python generate_dataset_csv.py
    ```
    This will create `user_event_interactions.csv`.

2.  **Train and Evaluate the Model**:
    Once the dataset is generated, run the main script:
    ```bash
    python train.py
    ```
    This script performs two main phases:
    *   **Training**: The model is trained on a large, combined graph of all historical interactions (`weeks 1-8`). A `LinkNeighborLoader` samples positive and negative user-item links and their computational subgraphs. The GNN is trained to discriminate between true and false links, learning powerful, general-purpose embeddings.
    *   **Evaluation**: The model's ability to predict future interactions is tested. It generates full GNN embeddings for each weekly snapshot in a historical window (`weeks 5-8`). These sequences of embeddings are fed into a Transformer to capture temporal dynamics. The model then predicts the Top-K items for users in the target week (`week 9`) and calculates Precision@K and Recall@K.

## Core Components

### 1. Data Processing (`data_utils.py`)
*   `load_interactions_from_csv()`: Reads the generated CSV, maps IP addresses and event IDs to unique integer IDs, and creates static random features for them.
*   `create_weekly_heterodata_from_df()`: Converts the interaction data for a single week into a `torch_geometric.data.HeteroData` object. It now creates both forward `('user', 'interacts_with', 'event')` and reverse `('event', 'rev_interacts_with', 'user')` edges to make the graph undirected for the sampler.

### 2. Model Architecture (`model.py`)

*   **`GNNEncoder(nn.Module)`**:
    *   Consists of multiple `SAGEConv` layers for both user and event nodes.
    *   Its `forward` method is designed to process mini-batches (sampled subgraphs) from a `LinkNeighborLoader`, performing message passing for a fixed number of hops.
    *   Includes an `inference` method that uses a `NeighborLoader` to efficiently compute embeddings for all nodes in a full graph, which is used during the evaluation phase.

*   **`PositionalEncoding(nn.Module)`**:
    *   Standard Transformer positional encoding, used during the temporal evaluation phase.

*   **`UserInteractionPredictor(nn.Module)`**: The main model orchestrating the two-phase process.
    *   **Training (`forward`)**: In the context of the new training loop, the main `forward` pass is now streamlined. It takes the pre-computed GNN embeddings for a sampled batch and passes them to a simple prediction head (`_compute_similarity_scores`) to calculate scores for positive and negative links.
    *   **Temporal Embedding Generation (`get_temporal_embeddings`)**: This method is used for evaluation. It takes a sequence of historical graph snapshots, uses the GNN's `inference` method on each to get full-graph embeddings, and then passes the sequence of user embeddings through the Transformer to get context-aware temporal embeddings.
    *   **Prediction (`predict_top_k_for_user`)**: Uses the final temporal user embedding and the final event embeddings to predict the Top-K most likely events for a given user.

### 3. Training and Evaluation (`train.py`)
*   **Constants**: Defines paths, model hyperparameters, training settings (like `BATCH_SIZE`), and sampling parameters (`NEIGHBORHOOD_SAMPLING_SIZES`).
*   **Training Setup**:
    *   Loads all weekly data.
    *   Combines the training snapshots (weeks 1-8) into a single, large `HeteroData` object.
    *   Initializes a `LinkNeighborLoader` on this combined graph. This loader manages sampling of positive/negative edges and their corresponding multi-hop neighborhoods.
*   **Training Loop**:
    *   Iterates for a fixed number of `EPOCHS` over the `LinkNeighborLoader`.
    *   In each step, it gets a mini-batch (sampled subgraph) from the loader.
    *   Passes the subgraph to `model.gnn_encoder` to get user and event embeddings.
    *   Calls the model's `forward` pass with these embeddings to get scores for the positive/negative links in the batch.
    *   Computes `BCEWithLogitsLoss` and updates the model parameters.
*   **Evaluation Logic**:
    *   After training, it proceeds to evaluation if enough data is available.
    *   Selects a historical window of snapshots (e.g., weeks 5-9) for the temporal model.
    *   Calls `model.get_temporal_embeddings` to run the full GNN+Transformer pipeline on this window of data.
    *   For each user in the target week, it calls `model.predict_top_k_for_user`.
    *   Calculates and prints average Precision@K and Recall@K against the ground truth of the target week.

## Key Architectural Change

The key change in this version is the **separation of the GNN training from the temporal training**.
1.  **GNN Training**: The GNN is trained first on a link prediction task over the entire training dataset. This allows it to learn powerful, static representations of nodes based on the overall graph structure, and this process is scalable thanks to neighborhood sampling.
2.  **Temporal Evaluation**: The pre-trained GNN is then used as a feature extractor. It generates high-quality embeddings for each temporal snapshot. The Transformer's role is to learn patterns from the *sequence* of these rich embeddings, which is a more focused and effective task than learning from raw features over time.