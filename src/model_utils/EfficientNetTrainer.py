import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

from src.models.EfficientNet import EfficientNetClassifier
from src.model_utils.EarlyStopping import EarlyStopping


class EfficientNetTrainer:
    def __init__(self,
                 model: EfficientNetClassifier,
                 dataloader: DataLoader,
                 optimizer,
                 loss,
                 valid_dataloader=None,
                 save_dir="checkpoints",
                 use_amp=False,
                 early_stopping_patience=5):

        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss = loss
        self.valid_dataloader = valid_dataloader

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.use_amp = use_amp
        self.scaler = torch.cuda.amp.GradScaler() if use_amp else None

        self.early_stopping = EarlyStopping(patience=early_stopping_patience,
                                            save_path=os.path.join(save_dir, "best_model.pt"))

        self.train_losses = []
        self.val_losses = []

    def train(self,
              epochs:int,
              valid:bool=True,
              plot:bool=True):

        self.model.to(self.device)

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for batch in self.dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss_value = self.loss(outputs, targets)
                    self.scaler.scale(loss_value).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(inputs)
                    loss_value = self.loss(outputs, targets)
                    loss_value.backward()
                    self.optimizer.step()

                running_loss += loss_value.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.dataloader.dataset)
            self.train_losses.append(epoch_loss)
            print(f"Epoch [{epoch + 1}/{epochs}] - Train Loss: {epoch_loss:.4f}")

            if valid and self.valid_dataloader is not None:
                val_loss, accuracy = self.validate()
                self.val_losses.append(val_loss)
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    break

        if plot:
            self.plot_losses()

    def validate(self):
        self.model.eval()
        val_loss: float = 0.0
        correct: int = 0
        total: int = 0

        with torch.no_grad():
            for batch in self.valid_dataloader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(inputs)
                        loss_value = self.loss(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    loss_value = self.loss(outputs, targets)

                val_loss += loss_value.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss /= len(self.valid_dataloader.dataset)
        accuracy = correct / total
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}")
        return val_loss, accuracy

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label="Train Loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.show()
