import torch
import torch.nn as nn
import torch.optim as optim
from TCN_dataloader import create_dataloaders

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        
        # padding='same' is crucial: it ensures our 374-minute length NEVER shrinks or grows.
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding='same', dilation=dilation)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 
                               padding='same', dilation=dilation)
        self.relu2 = nn.ReLU()
        
        # If the number of channels changes, we need a 1x1 conv to match dimensions for the skip connection
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()
            
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # Save the original input for the residual connection
        res = self.downsample(x)
        
        # Pass through the convolutions
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        
        # Add the residual back in (The "Skip Connection")
        return self.relu_out(out + res)


class MaskedTCN(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=[32, 64, 128], kernel_size=3):
        super().__init__()
        layers = []
        
        # Build the network by stacking TCN Blocks with exponentially increasing dilations
        for i in range(len(hidden_channels)):
            dilation_size = 2 ** i  # Dilations: 1, 2, 4
            in_ch = input_channels if i == 0 else hidden_channels[i-1]
            out_ch = hidden_channels[i]
            
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=dilation_size))
        
        self.tcn = nn.Sequential(*layers)
        
        # Final layer to squish the 128 channels back down to 1 channel (Close Returns)
        self.final_layer = nn.Conv1d(hidden_channels[-1], input_channels, kernel_size=1)

    def forward(self, x):
        # Input 'x' comes in as (Batch, 1, 374)
        features = self.tcn(x)
        
        # Output goes out as (Batch, 1, 374)
        return self.final_layer(features)
    
def init_weights(m):
    """
    Applies mathematically optimal starting weights for networks using ReLU.
    """
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # Initialize biases to zero
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def train_tcn(model, train_loader, val_loader, epochs=50, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training initialized on: {device}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- THE FIX: Switch from MSE to L1 (MAE) Loss ---
    # MAE doesn't square the error, so tiny mistakes still generate healthy gradients.
    criterion = nn.L1Loss(reduction='none')
    
    # --- THE FIX: Learning Rate Scheduler ---
    # If the validation loss doesn't improve for 3 epochs, cut the learning rate by 10x.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        total_train_loss = 0.0
        
        for masked_seq, true_seq, mask in train_loader:
            masked_seq = masked_seq.to(device)
            true_seq = true_seq.to(device)
            mask = mask.to(device)
            
            optimizer.zero_grad()
            predictions = model(masked_seq)
            hidden_mask = 1.0 - mask 
            
            raw_loss = criterion(predictions, true_seq)
            masked_loss = (raw_loss * hidden_mask).sum() / (hidden_mask.sum() + 1e-8)
            
            masked_loss.backward()
            optimizer.step()
            total_train_loss += masked_loss.item()
            
        # --- VALIDATION PHASE ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for masked_seq, true_seq, mask in val_loader:
                masked_seq, true_seq, mask = masked_seq.to(device), true_seq.to(device), mask.to(device)
                predictions = model(masked_seq)
                hidden_mask = 1.0 - mask
                
                raw_loss = criterion(predictions, true_seq)
                val_masked_loss = (raw_loss * hidden_mask).sum() / (hidden_mask.sum() + 1e-8)
                total_val_loss += val_masked_loss.item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        # --- THE FIX: Step the scheduler based on Validation Loss ---
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")

    print("Training Complete!")
    return model
# --- How to execute the pipeline ---
# Assuming you ran the create_dataloaders() function from the previous step:
train_loader, val_loader = create_dataloaders("processed_data/X_intraday.npy", batch_size=64)
# 
my_tcn = MaskedTCN(input_channels=1)
my_tcn.apply(init_weights)
trained_model = train_tcn(my_tcn, train_loader, val_loader, epochs=50)

# # Save the model weights
torch.save(trained_model.state_dict(), "tcn_self_supervised_weights.pth")
