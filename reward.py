import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np

import matplotlib.pyplot as plt

import argparse


class TrajectoryRewardNet(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        super(TrajectoryRewardNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.LayerNorm(hidden_size)
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
for param in net.parameters():
    # only apply to weights
    if len(param.shape) > 1:
        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


# Function to load data from .pkl file
def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    triples = [element[:3] for element in data]
    true_rewards = [element[-2:] for element in data]
    return triples, true_rewards


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
    try:
        for t1, t2, preference in data:
            t1_padded = pad_or_truncate(t1, max_length)
            t2_padded = pad_or_truncate(t2, max_length)

            # Flatten the list of tuples
            t1_flat = [item for sublist in t1_padded for item in sublist]
            t2_flat = [item for sublist in t2_padded for item in sublist]

            trajectories1.append(t1_flat)
            trajectories2.append(t2_flat)
            true_preferences.append(preference)
    except:
        import ipdb

        ipdb.set_trace()
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


def visualize_trajectories(batch_1, batch_2):
    for trajectory in batch_1:
        x = trajectory[::2]
        y = trajectory[1::2]
        plt.plot(x, y, color="red", alpha=0.1)
    for trajectory in batch_2:
        x = trajectory[::2]
        y = trajectory[1::2]
        plt.plot(x, y, color="blue", alpha=0.1)
    plt.show()


def train_model(file_path, epochs=1000, batch_size=32, model_path="best.pth"):
    data, true_rewards = load_data(file_path)
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
            try:
                loss = preference_loss(predicted_probabilities, batch_true_preferences)
                total_loss += loss.item()
            except:
                for p in net.parameters():
                    # get the smallest weight
                    min_weight = torch.min(p.data)
                    # get the largest weight
                    max_weight = torch.max(p.data)
                    # get the mean of the weights
                    mean_weight = torch.mean(p.data)
                    # check if any weights are nan
                    has_nan = torch.isnan(p.data).any()
                    # print everything
                    print(min_weight, max_weight, mean_weight, has_nan)
                visualize_trajectories(batch_trajectories1, batch_trajectories2)

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            for p in net.parameters():
                # get the smallest weight
                min_weight = torch.min(p.data)
                # get the largest weight
                max_weight = torch.max(p.data)
                # get the mean of the weights
                mean_weight = torch.mean(p.data)
                # check if any weights are nan
                has_nan = torch.isnan(p.data).any()
                # print everything
                print(min_weight, max_weight, mean_weight, has_nan)

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
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Prefereces"
    )
    parse.add_argument(
        "-d",
        "--database",
        type=str,
        help="Directory to trajectory database file",
    )
    parse.add_argument(
        "-e",
        "--epochs",
        type=int,
        help="Number of epochs to train the model",
    )
    args = parse.parse_args()
    if args.database:
        file_path = args.database
    else:
        file_path = "trajectories/database_350.pkl"
    if args.epochs:
        train_model(file_path, epochs=args.epochs)
    else:
        train_model(file_path, epochs=1000)
