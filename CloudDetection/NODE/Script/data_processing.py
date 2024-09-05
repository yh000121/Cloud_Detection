import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def load_and_preprocess_data(csv_path, batch_size=32, scaler_type='minmax'):
    """
    Loads the dataset from the provided CSV path, normalizes or standardizes the features,
    and splits the data into training and validation sets.

    :param csv_path: Path to the CSV file.
    :param batch_size: Batch size for DataLoader.
    :param scaler_type: Type of scaling ('minmax' for MinMaxScaler, 'standard' for StandardScaler).
    :return: DataLoader for training and validation data.
    """
    # Read the CSV file
    data = pd.read_csv(csv_path)

    # Separate features and labels (assuming first column is the label, and the rest are features)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values

    # Normalize or standardize features
    if scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Unsupported scaler_type. Choose either 'minmax' or 'standard'.")

    X = scaler.fit_transform(X)

    # Split the data into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert the data into PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Create DataLoader for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader
