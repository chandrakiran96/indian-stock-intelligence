import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 1. THE ARCHITECTURE (Copied from Phase 1)
# ==========================================
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same', dilation=dilation)
        self.relu2 = nn.ReLU()
        
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()
            
        self.relu_out = nn.ReLU()

    def forward(self, x):
        res = self.downsample(x)
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        return self.relu_out(out + res)

class MaskedTCN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=[32, 64, 128], kernel_size=3):
        super().__init__()
        layers = []
        for i in range(len(hidden_channels)):
            dilation_size = 2 ** i
            in_ch = input_channels if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=dilation_size))
        
        self.tcn = nn.Sequential(*layers)
        self.final_layer = nn.Conv1d(hidden_channels[-1], input_channels, kernel_size=1)

    def forward(self, x):
        features = self.tcn(x)
        return self.final_layer(features)

# ==========================================
# 2. THE EXTRACTION PIPELINE
# ==========================================
def extract_embeddings(intraday_path, weights_path, output_path, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Extraction initialized on: {device}")

    # A. Load the Raw Data and Scale it (Matching Phase 1)
    print("Loading intraday data...")
    X_intraday = np.load(intraday_path)
    X_tensor = torch.tensor(X_intraday, dtype=torch.float32) * 100.0
    
    # CRITICAL: shuffle=False ensures Day 1 embedding perfectly matches Day 1 overnight target
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # B. Load the Pre-Trained Model
    print("Loading pre-trained TCN weights...")
    model = MaskedTCN(input_channels=1)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.to(device)
    
    # CRITICAL: Set to eval mode to lock Dropout/BatchNorm layers
    model.eval()

    # C. Run Inference
    all_embeddings = []
    total_batches = len(loader)
    
    print("Extracting TCN Embeddings...")
    with torch.no_grad():  # Turn off gradients to save massive amounts of RAM
        for i, batch in enumerate(loader):
            X_batch = batch[0].to(device)
            
            # Pass through the core TCN layers
            features = model.tcn(X_batch) 
            
            # Slice the final minute which contains the causal memory of the day
            # features shape: (Batch, 128, 374) -> last_step shape: (Batch, 128)
            last_step = features[:, :, -1] 
            
            # Move back to CPU and convert to numpy
            all_embeddings.append(last_step.cpu().numpy())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total_batches} batches...")

    # D. Stitch together and Save
    print("Stitching arrays together...")
    final_embeddings = np.concatenate(all_embeddings, axis=0)
    
    assert final_embeddings.shape[0] == X_intraday.shape[0]
    print("Input rows :", X_intraday.shape[0])
    print("Output rows:", final_embeddings.shape[0])
    print(f"Final Embedding Shape: {final_embeddings.shape}")
    np.save(output_path, final_embeddings)
    print(f"Successfully saved to {output_path}")

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure these paths match your local directory structure
    extract_embeddings(
        intraday_path="processed_data/X_intraday.npy",   # Or wherever your file is
        weights_path="tcn_self_supervised_weights.pth",
        output_path="TCN_embeddings.npy"
    )