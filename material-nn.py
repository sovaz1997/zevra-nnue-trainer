import torch
import torch.nn as nn
import torch.optim as optim
from torch import float32
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class ChessDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, nrows=100000)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        inputs = row[[0, 1, 2, 3, 4, 5]].values.astype('float32')
        target = row[6].astype('float32')
        return torch.tensor(inputs), torch.tensor([target])


class LinearChessModel(nn.Module):
    def __init__(self):
        super(LinearChessModel, self).__init__()
        self.linear = nn.Linear(6, 256)
        self.inner = nn.Linear(256, 1)

    def forward(self, x):
        x = self.linear(x)
        x = self.inner(x)
        return x


def train_model(csv_file, num_epochs=1000, batch_size=64, learning_rate=0.01):
    dataset = ChessDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LinearChessModel()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    def print_evals(name, index):
        evals = []
        for st in range(0, 96):
            input = [0, 0, 0, 0, 0, st]
            input[index] = 1
            evals.append(model(torch.tensor(input, dtype=float32)).item())

        evals = reversed(evals)
        print(f"{name} evals: [{evals[0]}, {evals[1]}, ..., {evals[-1]}]")

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataset)
        scheduler.step(epoch_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")

        print_evals("Pawn", 0)
        print_evals("Knight", 1)
        print_evals("Bishop", 2)
        print_evals("Rook", 3)
        print_evals("Queen", 4)

    print("Training complete.")
    return model


if __name__ == "__main__":
    csv_file = "material-dataset.csv"
    trained_model = train_model(csv_file, num_epochs=1000, batch_size=512, learning_rate=0.1)
