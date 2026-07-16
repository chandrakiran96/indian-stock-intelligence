# NOTE: You must copy your TCNBlock and MaskedTCN classes from the previous script 
# into this file so the script knows what the architecture looks like!
import torch
import torch.nn as nn
import torch.optim as optim
from TCN_supervised_dataloader import create_finetune_dataloaders

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
            
class OvernightPredictor(nn.Module):
    def __init__(self, pretrained_tcn):
        super().__init__()
        self.feature_extractor = pretrained_tcn.tcn 
        
        # --- THE MLP HEAD FIX ---
        # Instead of 1 linear layer, we use a deep non-linear network
        self.predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout forces it to not rely on just one specific pattern
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)  
        last_step = features[:, :, -1]       # Slice the last minute (Batch, 128)
        output = self.predictor(last_step)   # Pass through the new MLP head
        return output


def finetune_model(model, train_loader, val_loader, epochs=30, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Fine-tuning initialized on: {device}")

    # --- THE DIFFERENTIAL LEARNING RATE FIX ---
    # Give the pre-trained TCN a tiny learning rate so it barely changes.
    # Give the new MLP head a normal learning rate so it learns fast.
    optimizer = optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': 1e-5},
        {'params': model.predictor.parameters(), 'lr': 1e-3}
    ])
    
    criterion = nn.L1Loss() 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # --- FIX 2: Force y_batch to have the exact same shape as predictions (Batch, 1) ---
            # This completely removes the broadcasting bug!
            y_batch = y_batch.view(-1, 1)
            
            optimizer.zero_grad()
            predictions = model(X_batch)
            
            # Now this computes a true 1-to-1 loss across your 64 samples
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                # Apply the same shape fix to the validation set
                y_batch = y_batch.view(-1, 1)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                total_val_loss += loss.item()
                
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] | Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")

    return model

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Initialize an empty TCN
    base_tcn = MaskedTCN(input_channels=1)
    
    # 2. Load the weights you just trained in Phase 1
    # Ensure the path matches where you saved your Phase 1 weights
    base_tcn.load_state_dict(torch.load("tcn_self_supervised_weights.pth", weights_only=True))

   
    
    # 3. Create the final Predictor model
    final_model = OvernightPredictor(base_tcn)
    
    # 4. Load Data & Train
    train_loader, val_loader = create_finetune_dataloaders("processed_data/X_intraday.npy", "processed_data/X_overnight.npy")
    trained_predictor = finetune_model(final_model, train_loader, val_loader)

    torch.save(trained_predictor.state_dict(), "tcn_overnight_predictor_weights.pth")
