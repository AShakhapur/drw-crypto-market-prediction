import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

### Warning don't accidentally pass in target in dim reduction!!!

class Autoencoder(nn.Module):

    def __init__(self, input_dim, encoding_dim):

        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return encoded, decoded
    
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

    def preprocessing(self, data, test_size = 0.2, batch_size=32):

        data_scaled = self.scaler.fit_transform(data)
        
        X_train, X_val = train_test_split(data_scaled, test_size=test_size, random_state=42)
        
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        
        train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader

    def train(self, train_loader, val_loader, epochs = 125, patience = 20, verbose = False):

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):

            self.model.train()
            train_loss = 0.0

            for batch_data, batch_target in train_loader:

                batch_data = batch_data.to(self.device)
                batch_target = batch_target.to(self.device)

                self.optimizer.zero_grad()
    
                outputs = self.model(batch_data)

                loss = self.criterion(outputs, batch_target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
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

            if verbose and epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'models/best_autoencoder.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        self.model.load_state_dict(torch.load('models/best_autoencoder.pth'))
        print(f'Training completed. Best validation loss: {best_val_loss:.6f}')
