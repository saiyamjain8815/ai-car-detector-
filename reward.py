import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
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


def bradley_terry_model(r1, r2):
    exp_r1 = torch.exp(r1)
    exp_r2 = torch.exp(r2)
    probability = exp_r1 / (exp_r1 + exp_r2)
    return probability.squeeze()


def preference_loss(predicted_probabilities, true_preferences):
    return F.binary_cross_entropy(predicted_probabilities, true_preferences)


input_size = 450 * 2
hidden_size = 128
learning_rate = 0.0003

net = TrajectoryRewardNet(input_size, hidden_size)
for param in net.parameters():
    if len(param.shape) > 1:
        nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain("relu"))
optimizer = optim.Adam(net.parameters(), lr=learning_rate)


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    triples = [element[:3] for element in data]
    true_rewards = [element[-2:] for element in data]
    return triples, true_rewards


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
        t1_flat = [item for sublist in t1_padded for item in sublist]
        t2_flat = [item for sublist in t2_padded for item in sublist]
        trajectories1.append(t1_flat)
        trajectories2.append(t2_flat)
        true_preferences.append(preference)
    trajectories1 = torch.tensor(trajectories1, dtype=torch.float32)
    trajectories2 = torch.tensor(trajectories2, dtype=torch.float32)
    true_preferences = torch.tensor(true_preferences, dtype=torch.float32)
    return trajectories1, trajectories2, true_preferences


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


def calculate_accuracy(predicted_probabilities, true_preferences):
    predicted_preferences = (predicted_probabilities > 0.5).float()
    correct_predictions = (predicted_preferences == true_preferences).float().sum()
    accuracy = correct_predictions / true_preferences.size(0)
    return accuracy.item()


def train_model(file_path, epochs=1000, batch_size=32, model_path="best.pth"):
    data, true_rewards = load_data(file_path)
    training_data, validation_data = train_test_split(
        data, test_size=0.2, random_state=42
    )
    trajectories1, trajectories2, true_preferences = prepare_data(training_data)
    validation_trajectories1, validation_trajectories2, validation_true_preferences = (
        prepare_data(validation_data)
    )

    dataset_size = len(true_preferences)
    validation_dataset_size = len(validation_true_preferences)

    if batch_size > dataset_size:
        batch_size = dataset_size

    best_loss = np.inf
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    for epoch in range(epochs):
        net.train()
        total_loss = 0.0
        total_accuracy = 0.0

        for i in range(0, dataset_size, batch_size):
            optimizer.zero_grad()

            batch_trajectories1 = trajectories1[i : i + batch_size]
            batch_trajectories2 = trajectories2[i : i + batch_size]
            batch_true_preferences = true_preferences[i : i + batch_size]

            rewards1 = net(batch_trajectories1)
            rewards2 = net(batch_trajectories2)

            predicted_probabilities = bradley_terry_model(rewards1, rewards2)

            loss = preference_loss(predicted_probabilities, batch_true_preferences)
            total_loss += loss.item()

            accuracy = calculate_accuracy(
                predicted_probabilities, batch_true_preferences
            )
            total_accuracy += accuracy * batch_true_preferences.size(0)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

        average_training_loss = total_loss / (dataset_size // batch_size)
        training_losses.append(average_training_loss)

        average_training_accuracy = total_accuracy / dataset_size
        training_accuracies.append(average_training_accuracy)

        net.eval()
        with torch.no_grad():
            validation_rewards1 = net(validation_trajectories1)
            validation_rewards2 = net(validation_trajectories2)
            validation_predicted_probabilities = bradley_terry_model(
                validation_rewards1, validation_rewards2
            )
            validation_loss = preference_loss(
                validation_predicted_probabilities, validation_true_preferences
            )
            validation_losses.append(validation_loss.item())

            validation_accuracy = calculate_accuracy(
                validation_predicted_probabilities, validation_true_preferences
            )
            validation_accuracies.append(validation_accuracy)

        if validation_loss.item() < best_loss:
            best_loss = validation_loss.item()
            torch.save(net.state_dict(), model_path)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {average_training_loss}, Val Loss: {validation_loss.item()}, Train Acc: {average_training_accuracy}, Val Acc: {validation_accuracy}"
            )

    plt.figure()
    plt.plot(training_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("figures/loss.png")

    plt.figure()
    plt.plot(training_accuracies, label="Train Accuracy")
    plt.plot(validation_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("figures/accuracy.png")


if __name__ == "__main__":
    parse = argparse.ArgumentParser(
        description="Training a Reward From Synthetic Preferences"
    )
    parse.add_argument(
        "-d", "--database", type=str, help="Directory to trajectory database file"
    )
    parse.add_argument(
        "-e", "--epochs", type=int, help="Number of epochs to train the model"
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
