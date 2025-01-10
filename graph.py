import matplotlib.pyplot as plt
import pandas as pd

def update_graph(csv_file, refresh_interval=5):
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    while True:
        data = pd.read_csv(csv_file)

        ax.clear()

        ax.plot(data["Epoch"], data["Train loss"], marker='o', label='Train Loss')
        ax.plot(data["Epoch"], data["Validate loss"], marker='o', label='Validate Loss', linestyle='--')

        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Train and Validate Loss Over Epochs')
        ax.legend()
        ax.grid(True)

        plt.pause(refresh_interval)

update_graph("trains/simple_deep/train.csv")
