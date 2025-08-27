import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pyarrow

### Warning don't accidentally pass in target in dim reduction!!!

class Autoencoder(nn.Module):

    def __init__(self, input_dim, encoding_dim):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim * 0.7)),  # ~65% of input dim
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_dim * 0.7), int(input_dim * 0.4)),  # ~33%
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_dim * 0.4), int(input_dim * 0.2)),  # ~16%
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_dim * 0.2), encoding_dim),
            nn.LeakyReLU(0.1)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, int(input_dim * 0.2)),  # mirror encoder
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_dim * 0.2), int(input_dim * 0.4)),
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_dim * 0.4), int(input_dim * 0.7)),
            nn.LeakyReLU(0.1),
            nn.Linear(int(input_dim * 0.7), input_dim)
        )



    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded  # Only return decoded, not tuple
    
    def encode(self, x):
        return self.encoder(x)
    

class AutoencoderTrainer:

    def __init__(self, input_dim, encoding_dim, learning_rate=0.001):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_dim, encoding_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scaler = StandardScaler()
        self.train_losses = []
        self.val_losses = []

    def preprocessing(self, data, test_size=0.2, batch_size=64):
        
        n_samples = len(data)
        split_idx = int(n_samples * (1 - test_size))

        # Scale entire dataset (fit only on train portion to avoid leakage)
        self.scaler.fit(data[:split_idx])
        data_scaled = self.scaler.transform(data)

        # Time-based split
        X_train, X_val = data_scaled[:split_idx], data_scaled[split_idx:]

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)

        # Datasets
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, X_val_tensor)

        # Shuffle within subsets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # no shuffle for eval

        return train_loader, val_loader

    def train(self, train_loader, val_loader, epochs = 125, patience = 20, verbose = True):

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):

            self.model.train()
            train_loss = 0.0

            for batch_data, batch_target in train_loader:

                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)

                self.optimizer.zero_grad()
    
                outputs = self.model(batch_data)  # Now returns only decoded

                loss = self.criterion(outputs, batch_target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                # Fix: Use consistent batch unpacking here too
                for batch_data, batch_target in val_loader:
                    batch_data = batch_data.to(self.device)
                    batch_target = batch_target.to(self.device)
                    
                    outputs = self.model(batch_data)
                    loss = self.criterion(outputs, batch_target)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

            if verbose:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
                    # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f'../src/models/best_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # reload best weights
        self.model.load_state_dict(torch.load(f'../src/models/best_autoencoder.pth'))
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")


        self.model.load_state_dict(torch.load('../src/models/best_autoencoder.pth', weights_only=True))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')

    def encode_data(self, data):
        # Encodes the data 

        self.model.eval()
        
        data_scaled = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_scaled).to(self.device)
        
        with torch.no_grad():
            encoded = self.model.encode(data_tensor)
            
        return encoded.cpu().numpy()
    
    def reconstruct_data(self, data):
        data_scaled = self.scaler.transform(data)
        data_tensor = torch.FloatTensor(data_scaled).to(self.device)
        with torch.no_grad():
            reconstructed = self.model(data_tensor)
        return reconstructed.cpu().numpy()

    def get_reconstruction_error(self, data):
        # Calc reconstruction error

        original = data
        reconstructed = self.reconstruct_data(data)
        
        mse = np.mean((original - reconstructed) ** 2, axis=1)
        return mse
    
    def plot_losses(self):
        # Plotting losses

        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

def autoencoder_workflow(train_df, encoding_dim=None, epochs=100):
    """
    Complete workflow for autoencoder dimensionality reduction
    
    Args:
        train_df: pandas DataFrame with features
        encoding_dim: dimensionality of encoded features (default: input_dim // 4)
        epochs: number of training epochs
    
    Returns:
        trainer: trained AutoencoderTrainer object
        encoded_features: numpy array of encoded features
    """
    
    # Prepare parameters
    input_dim = train_df.shape[1]
    if encoding_dim is None:
        encoding_dim = max(2, input_dim // 4)  # Default to 1/4 of input dimension
    
    print(f"Input dimension: {input_dim}")
    print(f"Encoding dimension: {encoding_dim}")
    print(f"Compression ratio: {input_dim/encoding_dim:.2f}x")
    
    # Initialize trainer
    trainer = AutoencoderTrainer(input_dim, encoding_dim)
    
    # Prepare data
    train_loader, val_loader = trainer.preprocessing(train_df.values)

    # Train the model
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=epochs)
    trainer.plot_losses()
    
    # Extract encoded features
    print("Extracting encoded features...")
    encoded_features = trainer.encode_data(train_df.values)
    
    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=[f'encoded_feature_{i}' for i in range(encoding_dim)],
        index=train_df.index
    )
    
    print(f"Original shape: {train_df.shape}")
    print(f"Encoded shape: {encoded_df.shape}")
    
    return trainer, encoded_df


#train_df = pd.read_parquet("../data/train.parquet")
#autoencoder_workflow(train_df=train_df, epochs=20)
