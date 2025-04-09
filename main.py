import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, r2_score
from tabnet import TabNet
from dataset import TabularDataset
from data_loader import load_data

def main():
    # Define file paths for input features and target data
    file_paths = ['nutritionalgit.xlsx', 'physiologicalgit.xlsx', 'genetics.xlsx', 'lifestylegit.xlsx']

    # Load the features and target labels
    features_list, labels = load_data(file_paths, target_file='target.xlsx')

    # Prepare the dataset
    full_dataset = TabularDataset(features_list, labels)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create DataLoaders for training and testing
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model initialization
    feature_dims = [f.shape[1] for f in features_list]  # Get number of features for each modality
    model = TabNet(feature_dims)

    # Set up loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=2e-2)

    # Training and evaluation
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer)

def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer):
    # Initialize variables to store metrics
    mae_list = []
    r2_list = []

    # Training loop (200 epochs)
    for epoch in range(200):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Evaluation
        model.eval()
        predictions, true_labels = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                predictions.extend(outputs.squeeze().tolist())
                true_labels.extend(labels.squeeze().tolist())

        # Calculate MAE and R^2
        mae = mean_absolute_error(true_labels, predictions)
        r2 = r2_score(true_labels, predictions)

        mae_list.append(mae)
        r2_list.append(r2)

        print(f"Epoch {epoch + 1}/200")
        print(f"MAE: {mae:.4f}")
        print(f"R^2: {r2:.4f}")
        print('-' * 30)

    # Calculate the mean scores across all epochs
    mean_mae = sum(mae_list) / len(mae_list)
    mean_r2 = sum(r2_list) / len(r2_list)

    print(f"Mean MAE over all epochs: {mean_mae:.4f}")
    print(f"Mean R^2 over all epochs: {mean_r2:.4f}")

if __name__ == "__main__":
    main()
