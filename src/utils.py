import matplotlib.pyplot as plt

def plot_neural(epochs: int, train_loss: list, val_loss: list):

    plt.plot(
        range(epochs), train_loss, label='Training Loss', linewidth=2.0, color='blue', ls='dashed'
    )
    plt.plot(
        range(epochs), val_loss, label='Val Loss', linewidth=2.0, color='green', ls='solid'
    )
    plt.plot(
        range(epochs), [min(val_loss)]*epochs, label='Min Val Loss', linewidth=0.8, color='black', ls='dotted'
    )
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(axis='x')