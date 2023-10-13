from NevoboResultParser import MatchResult, get_historic_results

# Step 1: Load and preprocess the data
# Assuming 'data' is a list of dictionaries with MatchResult objects
data = get_historic_results()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd
import numpy as np

# Assuming 'data' is a flat array of MatchResult objects

# Extract 'outcome' values from MatchResult objects
outcomes = [match.outcome for match in data]

# Define and extract categorical features from MatchResult objects
team_home = [match.team_home for match in data]
team_away = [match.team_away for match in data]
region = [match.region for match in data]

# Encode the 'outcome' labels
label_encoder = LabelEncoder()
encoded_outcomes = label_encoder.fit_transform(outcomes)

# Perform one-hot encoding for categorical features
onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

# Encode and concatenate the categorical features
team_home_encoded = onehot_encoder.fit_transform(np.array(team_home).reshape(-1, 1))
team_away_encoded = onehot_encoder.fit_transform(np.array(team_away).reshape(-1, 1))

# Combine one-hot encoded features
features = np.concatenate((team_home_encoded, team_away_encoded), axis=1)

# Split the data into training and validation sets
features_train, features_val, labels_train, labels_val = train_test_split(
    features, encoded_outcomes, test_size=0.2, random_state=42
)

# Define a custom dataset
class MatchResultDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx])

# Rest of the code remains the same...


# Define the neural network architecture
class OutcomePredictionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(OutcomePredictionModel, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# Create DataLoader for training
train_dataset = MatchResultDataset(features_train, labels_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create DataLoader for validation
val_dataset = MatchResultDataset(features_val, labels_val)
val_loader = DataLoader(val_dataset, batch_size=32)

# Define loss function and optimizer
input_dim = len(label_encoder.classes_)  # Number of unique outcome classes
model = OutcomePredictionModel(input_dim=input_dim, output_dim=input_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Evaluation function
def evaluate_model(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Evaluate the model
evaluate_model(model, val_loader)

# Make predictions function
def predict_outcome(model, input_features):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(input_features).float()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return label_encoder.inverse_transform(predicted)

# Example usage
input_features = [[feature1, feature2, feature3]]  # Replace with your input features
predicted_outcome = predict_outcome(model, input_features)
print(f"Predicted Outcome: {predicted_outcome[0]}")
