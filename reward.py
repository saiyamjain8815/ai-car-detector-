import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np

import matplotlib.pyplot as plt


class TrajectoryRewardNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(TrajectoryRewardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# Define the Bradley-Terry model to compute preferences
def bradley_terry_model(r1, r2):
    exp_r1 = torch.exp(r1)
    exp_r2 = torch.exp(r2)
    probability = exp_r1 / (exp_r1 + exp_r2)
    return probability.squeeze()


# Define the loss function
def preference_loss(predicted_probabilities, true_preferences):
    return F.binary_cross_entropy(predicted_probabilities, true_preferences)


# Sample input size and hyperparameters
input_size = 450 * 2  # Length of trajectory * 2 (for x and y points)
hidden_size = 128
learning_rate = 0.0003

# Instantiate the network, optimizer, and criterion
net = TrajectoryRewardNet(input_size, hidden_size)
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# Function to load data from .pkl file
def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


# Function to prepare data for training
def prepare_data(data, max_length=input_size // 2):
    def pad_or_truncate(trajectory, length):
        if len(trajectory) > length:
            return trajectory[:length]
        else:
            padding = [trajectory[-1]] * (length - len(trajectory))
            return trajectory + padding

    trajectories1 = []
    trajectories2 = []
    true_preferences = []

    for t1, t2, preference in data:
        t1_padded = pad_or_truncate(t1, max_length)
        t2_padded = pad_or_truncate(t2, max_length)

        # Flatten the list of tuples
        t1_flat = [item for sublist in t1_padded for item in sublist]
        t2_flat = [item for sublist in t2_padded for item in sublist]

        trajectories1.append(t1_flat)
        trajectories2.append(t2_flat)
        true_preferences.append(preference)

    trajectories1 = torch.tensor(trajectories1, dtype=torch.float32)
    trajectories2 = torch.tensor(trajectories2, dtype=torch.float32)
    true_preferences = torch.tensor(true_preferences, dtype=torch.float32)

    return trajectories1, trajectories2, true_preferences


def prepare_single_trajectory(trajectory, max_length=input_size // 2):
    def pad_or_truncate(trajectory, length):
        if len(trajectory) > length:
            return trajectory[:length]
        else:
            padding = [trajectory[-1]] * (length - len(trajectory))
            return trajectory + padding

    # Pad or truncate the trajectory
    trajectory_padded = pad_or_truncate(trajectory, max_length)
    # Flatten the list of tuples
    trajectory_flat = [item for sublist in trajectory_padded for item in sublist]

    # Convert to tensor and add an extra dimension
    trajectory_tensor = torch.tensor([trajectory_flat], dtype=torch.float32)

    return trajectory_tensor


def train_model(file_path, epochs=1000, batch_size=32, model_path="best.pth"):
    data = load_data(file_path)
    trajectories1, trajectories2, true_preferences = prepare_data(data)

    dataset_size = len(true_preferences)

    if batch_size > dataset_size:
        batch_size = dataset_size

    best_loss = np.inf

    network_reward_comparator = []
    with open("trajectories/comparator.pkl", "rb") as f:
        comparator_reward, comparator_trajectory = pickle.load(f)
    comparator_trajectory = prepare_single_trajectory(comparator_trajectory)

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0

        for i in range(0, dataset_size, batch_size):
            optimizer.zero_grad()

            batch_trajectories1 = trajectories1[i : i + batch_size]
            batch_trajectories2 = trajectories2[i : i + batch_size]
            batch_true_preferences = true_preferences[i : i + batch_size]

            # Forward pass for both trajectories
            rewards1 = net(batch_trajectories1)
            rewards2 = net(batch_trajectories2)

            # Compute preference probabilityabilities
            predicted_probabilities = bradley_terry_model(rewards1, rewards2)

            # Compute loss
            loss = preference_loss(predicted_probabilities, batch_true_preferences)
            total_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

        average_loss = total_loss / (dataset_size // batch_size)

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(net.state_dict(), model_path)
            # print(f"Model saved with loss: {best_loss}")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {average_loss}")

        net.eval()
        network_reward_comparator.append(net(comparator_trajectory).item())

    plt.plot(network_reward_comparator, color="blue", label="Network Reward")
    plt.plot(
        [comparator_reward] * len(network_reward_comparator),
        color="red",
        label="True Reward",
    )
    plt.legend()
    plt.savefig("figures/reward.png")


if __name__ == "__main__":
    file_path = "trajectories/database_50.pkl"
    train_model(file_path)
