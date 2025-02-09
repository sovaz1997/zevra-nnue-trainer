import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model.nnue import NNUE
from src.networks.simple.data_manager import SCALE


def sigmoid(cp_space_eval: torch.Tensor, scale_factor = 400.0):
    return torch.sigmoid(cp_space_eval / scale_factor)



def compute_loss(loss: float):
    return loss * SCALE * SCALE

def validate(model: nn.Module, validation_dataloader: DataLoader):
    batches_length = 0
    criterion = nn.MSELoss()
    running_loss = 0.0

    for batch_idx, (batch_inputs, batch_scores) in enumerate(validation_dataloader):
        batches_length += 1
        for i, batch_input in enumerate(batch_inputs):
            batch_inputs[i] = batch_inputs[i].to("mps")
        batch_scores = batch_scores.to("mps")
        outputs = sigmoid(model(*batch_inputs))
        loss = criterion(outputs.squeeze(), sigmoid(batch_scores))
        running_loss += loss.item()

    return compute_loss(running_loss / batches_length)

def save_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        train_directory,
):
    checkpoint = {
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, train_directory + "/checkpoint.pth")
    model.save_weights(epoch, train_directory)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(
        model,
        optimizer,
        scheduler,
        train_directory,):
    filename = train_directory + "/checkpoint.pth"
    if not os.path.exists(filename):
        return 0
    checkpoint = torch.load(filename, weights_only=True)
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    model.load_weights(epoch, train_directory)

    return epoch


def train(
        model: NNUE,
        train_data_loader: DataLoader,
        validation_data_loader: DataLoader,
        train_directory: str
):
    if not os.path.exists(train_directory):
        os.makedirs(train_directory)

    device = torch.device("mps")
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    epoch = load_checkpoint(model, optimizer, scheduler, train_directory) + 1

    # print(f'Validate: {validate(model, validation_data_loader)}')
    # return None


    TRAIN_FILE = f'{train_directory}/train.csv'
    with open(TRAIN_FILE, 'a') as train:
        train.write('Epoch,Train loss,Validate loss\n')

    while True:
        model.train()
        running_loss = 0.0
        count = 0
        index = 0

        for batch_idx, (batch_inputs, batch_scores) in enumerate(train_data_loader):
            index += 1
            if index % 1000 == 0:
                print(f"Learning: {index}", flush=True)
            count += len(batch_inputs[0])

            for i, batch_input in enumerate(batch_inputs):
                batch_inputs[i] = batch_inputs[i].to(device, non_blocking=True)

            batch_scores = batch_scores.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = sigmoid(model(*batch_inputs))
            loss = criterion(outputs.squeeze(), sigmoid(batch_scores))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        loss = (running_loss / index)
        scheduler.step(loss)

        validate_loss = validate(model, validation_data_loader)
        print_loss = compute_loss(loss)
        print(f"Epoch [{epoch}], Train loss: {print_loss:.12f}, Validate loss: {validate_loss:.4f}", flush=True)
        save_checkpoint(model, optimizer, scheduler, epoch, train_directory)

        with open(TRAIN_FILE, 'a') as train:
            train.write(f"{epoch},{print_loss:.12f},{validate_loss:.12f}\n")

        epoch += 1

        # if loss < 0.05:
        #     break
        print(optimizer.param_groups[0]['lr'])