import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MaskedStockDataset(Dataset):
    def __init__(self, intraday_path, mask_ratio=0.15):
        """
        Loads the clean numpy arrays and prepares them for dynamic masking.
        """
        print("Loading data into RAM...")
        # Load the saved numpy array (Shape: Total Profiles, 1, 374)
        self.data = np.load(intraday_path)
        
        # Convert to PyTorch Tensors (Float32 is standard for neural networks)
        self.data = torch.tensor(self.data, dtype=torch.float32)*100.0
        self.mask_ratio = mask_ratio
        
        print(f"Successfully loaded {len(self.data)} trading days.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        This function runs dynamically every time PyTorch grabs a day of data.
        It returns three things: the masked input, the original target, and the mask itself.
        """
        # 1. Grab the clean true sequence: Shape (1, 374)
        true_sequence = self.data[idx]
        
        # 2. Create a random mask of 0s and 1s of the exact same shape
        # PyTorch generates random probabilities between 0 and 1. 
        # If the probability is greater than 0.15, it stays 1 (Visible). If less, it becomes 0 (Masked).
        rand_matrix = torch.rand(true_sequence.shape)
        mask = (rand_matrix > self.mask_ratio).float()
        
        # 3. Apply the mask to create the corrupted input
        # Multiplying by the mask turns 15% of the values to exactly 0.0
        masked_sequence = true_sequence * mask
        
        # 4. We return all three so the loss function knows which parts were hidden
        return masked_sequence, true_sequence, mask


# --- HOW TO USE IT FOR TRAINING ---
def create_dataloaders(intraday_npy_path, batch_size=64):
    # Initialize the dataset
    dataset = MaskedStockDataset(intraday_npy_path, mask_ratio=0.15)
    
    # Split 80% for Training, 20% for Validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create the DataLoaders (this handles batching and shuffling automatically)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Example Usage:
# train_loader, val_loader = create_dataloaders("processed_data/X_intraday.npy", batch_size=64)