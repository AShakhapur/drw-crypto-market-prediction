import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Autoencoder(nn.Module):
    def __init__(self, input_dim=20, encoding_dim=8):
        super().__init__()
        hidden1 = int(input_dim * 0.8)
        hidden2 = int(input_dim * 0.5)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden2, encoding_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, hidden2),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden2, hidden1),
            nn.LeakyReLU(0.01),
            nn.Linear(hidden1, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderTrainer:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.001):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_dim, encoding_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scaler = StandardScaler()
        self.train_losses = []
        self.val_losses = []

    def preprocessing(self, data, test_size=0.2, batch_size=256):
        n_samples = len(data)
        split_idx = int(n_samples * (1 - test_size))
        self.scaler.fit(data[:split_idx])
        data_scaled = self.scaler.transform(data)
        X_train, X_val = data_scaled[:split_idx], data_scaled[split_idx:]
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train)),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(X_val)),
                                batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    def train(self, train_loader, val_loader, epochs=125, patience=20, verbose=True):
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_data, batch_target in train_loader:
                batch_data, batch_target = batch_data.to(self.device), batch_target.to(self.device)
                noisy_data = batch_data + 0.03 * torch.randn_like(batch_data)
                self.optimizer.zero_grad()
                outputs = self.model(noisy_data)
                loss = self.criterion(outputs, batch_target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_data, batch_target in val_loader:
                    batch_data, batch_target = batch_data.to(self.device), batch_target.to(self.device)
                    outputs = self.model(batch_data)
                    val_loss += self.criterion(outputs, batch_target).item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)

            if verbose:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), f'../src/models/best_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        self.model.load_state_dict(torch.load(f'../src/models/best_autoencoder.pth'))
        print(f"Training completed. Best validation loss: {best_val_loss:.6f}")

    def encode_data(self, data):
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
        original = data
        reconstructed = self.reconstruct_data(data)
        return np.mean((original - reconstructed) ** 2, axis=1)

    def plot_losses(self):
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
    input_dim = train_df.shape[1]
    if encoding_dim is None:
        encoding_dim = max(2, input_dim // 4)

    print(f"Input dimension: {input_dim}")
    print(f"Encoding dimension: {encoding_dim}")
    print(f"Compression ratio: {input_dim/encoding_dim:.2f}x")
    print(f"Loss Function: MSE")

    trainer = AutoencoderTrainer(input_dim, encoding_dim)
    train_loader, val_loader = trainer.preprocessing(train_df.values)
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=epochs)
    trainer.plot_losses()

    encoded_features = trainer.encode_data(train_df.values)
    encoded_df = pd.DataFrame(encoded_features,
                              columns=[f'encoded_feature_{i}' for i in range(encoding_dim)],
                              index=train_df.index)
    reconstruction_error = trainer.get_reconstruction_error(train_df.values).mean()

    print(f"Original shape: {train_df.shape}")
    print(f"Encoded shape: {encoded_df.shape}")
    print(f"Mean reconstruction error: {reconstruction_error:.6f}")

    return trainer, encoded_df, reconstruction_error