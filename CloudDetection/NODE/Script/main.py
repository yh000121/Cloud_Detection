# main.py

import torch
from model import AdvancedNODE  # Import the model definition
from train import train_model  # Import the training function
from data_processing import load_and_preprocess_data  # Import the data loading function


def main():
    # CSV file path
    csv_path = r'D:\CloudDetection\NODE\data\balanced_merged_features.csv'

    # Set data loader parameters
    batch_size = 64  # Adjust batch size based on your system
    scaler_type = 'minmax'  # Choose 'minmax' or 'standard' for scaling

    # Load and preprocess data
    train_loader, val_loader = load_and_preprocess_data(csv_path, batch_size=batch_size, scaler_type=scaler_type)

    # Model parameters
    num_trees = 100  # Number of trees
    depth = 10  # Depth of each tree
    num_features = next(iter(train_loader))[0].shape[1]  # Get number of features dynamically
    num_classes = 3  # Number of classes for classification
    dropout_rate = 0.3  # Dropout rate for regularization
    l2_lambda = 0.01  # L2 regularization strength

    # Initialize the model
    model = AdvancedNODE(num_trees=num_trees, num_features=num_features, num_classes=num_classes,
                         depth=depth, dropout_rate=dropout_rate, l2_lambda=l2_lambda)

    # Training parameters
    epochs = 30
    learning_rate = 0.001

    # Start training the model
    print("Training the model...")
    train_model(model, train_loader, val_loader, epochs=epochs, lr=learning_rate)


if __name__ == "__main__":
    main()
