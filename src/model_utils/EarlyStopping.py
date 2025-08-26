import torch
import torch.nn as nn

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self,
                 patience: int = 5,
                 verbose: bool = True,
                 delta: float = 0.0,
                 save_path: str = "checkpoint.pt"):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            verbose (bool): Print messages when improvement occurs.
            delta (float): Minimum change to qualify as improvement.
            save_path (str): Path to save the best model.
        """
        self.patience: int = patience
        self.verbose: bool = verbose
        self.delta: float = delta
        self.save_path: str = save_path
        self.best_loss: float = float('inf')
        self.counter: int = 0
        self.early_stop: bool = False

    def __call__(self, val_loss:float, model:nn.Module):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
            if self.verbose:
                print(f"Validation loss improved. Saving model to {self.save_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss for {self.counter} epochs.")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered!")
