import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SupervisedStockDataset(Dataset):
    def __init__(self, intraday_path, overnight_path):
        print("Loading supervised data into RAM...")
        # Load both arrays
        intraday_data = np.load(intraday_path)
        overnight_data = np.load(overnight_path)
        
        # Scale both by 100 to match pre-training logic
        self.X = torch.tensor(intraday_data, dtype=torch.float32) * 100.0
        self.y = torch.tensor(overnight_data, dtype=torch.float32) * 100.0
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_finetune_dataloaders(intraday_path, overnight_path, batch_size=64):
    dataset = SupervisedStockDataset(intraday_path, overnight_path)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader