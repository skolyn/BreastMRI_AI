import os

import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from src.model_utils.BreastMRIDataset import BreastMRIDataset
from src.model_utils.EfficientNetTrainer import EfficientNetTrainer
from src.models.EfficientNet import EfficientNetClassifier

labels_df = pd.read_csv("../../data/fastMRI_breast_labels/fastMRI_breast_modified_labels.csv")
labels_df.columns = labels_df.columns.str.strip()
labels_df = labels_df.rename(columns={
    'Lesion status (0 = negative, 1= malignancy, 2= benign)': 'Lesion_status',
    'Data split (0=training, 1=testing)': 'Data_split',
    'Patient Coded Name': 'Patient'
})

train_df = labels_df[labels_df["Data_split"] == 0]
val_df = labels_df[labels_df["Data_split"] == 1]

train_labels = dict(zip(train_df["Patient"], train_df["Lesion_status"]))
val_labels = dict(zip(val_df["Patient"], val_df["Lesion_status"]))

data_root = "../../data/injected"
all_cases = set(os.listdir(data_root))

train_cases = [c for c in train_labels.keys() if c in all_cases]
val_cases = [c for c in val_labels.keys() if c in all_cases]

train_dataset = BreastMRIDataset(data_root, train_cases, train_labels)
valid_dataset = BreastMRIDataset(data_root, val_cases, val_labels)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=4, shuffle=True)

model = EfficientNetClassifier()
optimizer = AdamW(model.parameters(), lr=1e-3)
criterion = CrossEntropyLoss()

trainer = EfficientNetTrainer(model=model,
                              dataloader=train_dataloader,
                              optimizer=optimizer,
                              loss=criterion,
                              valid_dataloader=valid_dataloader,
                              save_dir="efficientnet_checkpoints")

trainer.train(100)
