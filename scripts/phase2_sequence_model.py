#!/usr/bin/env python3
"""
Phase 2: Sequence Model Training Framework
LSTM-based full sequence analysis for Cup & Handle patterns
Target: 76% success rate (based on research)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import json
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from datetime import datetime

class SequenceDataset(Dataset):
    """
    Dataset for full sequence OHLCV data with Cup & Handle labels
    """
    
    def __init__(self, sequences, labels, sequence_length=400):
        self.sequences = sequences
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        label = torch.LongTensor([self.labels[idx]])
        return sequence, label

class FullSequenceModel(nn.Module):
    """
    Full Sequence LSTM Model for Cup & Handle Pattern Detection
    Research Target: 76% accuracy
    """
    
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, num_classes=3, dropout=0.3):
        super(FullSequenceModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,  # bidirectional
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)  # [Buy, Hold, Sell]
        )
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param.data)
                else:
                    nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last output for classification
        final_output = attn_out[:, -1, :]
        
        # Classification
        output = self.classifier(final_output)
        
        return output

class TechnicalIndicatorModel(nn.Module):
    """
    Technical Indicator Model for Cup & Handle validation
    Research Target: 70% accuracy
    """
    
    def __init__(self, input_size=10, hidden_size=32, num_classes=3, dropout=0.2):
        super(TechnicalIndicatorModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, num_classes)
        )
    
    def forward(self, x):
        return self.feature_extractor(x)

class PatternSequenceProcessor:
    """
    Process raw pattern data into sequences for model training
    """
    
    def __init__(self, sequence_length=400):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def load_pattern_data(self, data_dir='data/raw/crypto_patterns'):
        """Load and process pattern data from collected files"""
        data_dir = Path(data_dir)
        
        sequences = []
        labels = []
        
        # Load crypto data files
        for csv_file in data_dir.glob('*_4h_*.csv'):
            print(f"Processing {csv_file.name}...")
            
            # Load OHLCV data
            data = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # Load corresponding patterns
            pattern_file = csv_file.stem + '_pattern_candidates.json'
            pattern_path = data_dir / f"{pattern_file}"
            
            if pattern_path.exists():
                with open(pattern_path, 'r') as f:
                    patterns = json.load(f)
                
                # Process each pattern
                for pattern in patterns[:100]:  # Limit for initial training
                    seq, label = self.extract_pattern_sequence(data, pattern)
                    if seq is not None:
                        sequences.append(seq)
                        labels.append(label)
        
        print(f"Processed {len(sequences)} pattern sequences")
        return np.array(sequences), np.array(labels)
    
    def extract_pattern_sequence(self, data, pattern):
        """Extract standardized sequence from pattern data"""
        try:
            start_idx = pattern['start_idx']
            end_idx = pattern['end_idx']
            
            # Ensure we have enough data
            if end_idx - start_idx < 50:
                return None, None
            
            # Extract sequence data
            sequence_data = data.iloc[start_idx:end_idx]
            
            # Create OHLCV features
            features = []
            for _, row in sequence_data.iterrows():
                ohlcv = [
                    row['Open'],
                    row['High'], 
                    row['Low'],
                    row['Close'],
                    row['Volume']
                ]
                features.append(ohlcv)
            
            # Normalize to fixed length
            features = np.array(features)
            if len(features) > self.sequence_length:
                # Sample evenly
                indices = np.linspace(0, len(features)-1, self.sequence_length, dtype=int)
                features = features[indices]
            elif len(features) < self.sequence_length:
                # Pad with last values
                padding = np.repeat(features[-1:], self.sequence_length - len(features), axis=0)
                features = np.vstack([features, padding])
            
            # Generate label based on pattern success
            total_return = pattern['price_action']['total_return']
            if total_return > 0.05:  # 5%+ return = Buy signal
                label = 0  # Buy
            elif total_return < -0.05:  # -5% return = Sell signal
                label = 2  # Sell
            else:
                label = 1  # Hold
            
            return features, label
            
        except Exception as e:
            print(f"Error processing pattern: {e}")
            return None, None
    
    def create_technical_indicators(self, data):
        """Create technical indicator features"""
        indicators = {}
        
        # RSI
        if 'RSI' in data.columns:
            indicators['RSI'] = data['RSI'].iloc[-1]
        
        # Moving averages
        if 'SMA_20' in data.columns and 'SMA_50' in data.columns:
            indicators['MA_ratio'] = data['SMA_20'].iloc[-1] / data['SMA_50'].iloc[-1]
        
        # Bollinger Band position
        if 'BB_Position' in data.columns:
            indicators['BB_position'] = data['BB_Position'].iloc[-1]
        
        # Volume ratio
        if 'Volume_Ratio' in data.columns:
            indicators['Volume_ratio'] = data['Volume_Ratio'].iloc[-1]
        
        # Price momentum
        if len(data) >= 10:
            recent_return = (data['Close'].iloc[-1] - data['Close'].iloc[-10]) / data['Close'].iloc[-10]
            indicators['Price_momentum'] = recent_return
        
        # Volatility
        if 'Volatility_10' in data.columns:
            indicators['Volatility'] = data['Volatility_10'].iloc[-1]
        
        # Support/Resistance
        if all(col in data.columns for col in ['Support_Level', 'Resistance_Level']):
            current_price = data['Close'].iloc[-1]
            support = data['Support_Level'].iloc[-1]
            resistance = data['Resistance_Level'].iloc[-1]
            
            indicators['Support_distance'] = (current_price - support) / support
            indicators['Resistance_distance'] = (resistance - current_price) / current_price
        
        # Fill missing indicators with defaults
        indicator_names = ['RSI', 'MA_ratio', 'BB_position', 'Volume_ratio', 
                          'Price_momentum', 'Volatility', 'Support_distance', 
                          'Resistance_distance', 'MACD_signal', 'ADX']
        
        feature_vector = []
        for name in indicator_names:
            feature_vector.append(indicators.get(name, 0.5))  # Default to neutral value
        
        return np.array(feature_vector)

class ModelTrainer:
    """
    Train and validate the sequence models
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print(f"Using device: {device}")
        
    def train_sequence_model(self, sequences, labels, epochs=50, batch_size=32):
        """Train the full sequence LSTM model"""
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, labels, test_size=0.2, stratify=labels, random_state=42
        )
        
        # Normalize sequences
        scaler = StandardScaler()
        X_train_scaled = []
        X_test_scaled = []
        
        for seq in X_train:
            seq_scaled = scaler.fit_transform(seq)
            X_train_scaled.append(seq_scaled)
        
        for seq in X_test:
            seq_scaled = scaler.transform(seq)
            X_test_scaled.append(seq_scaled)
        
        # Create datasets
        train_dataset = SequenceDataset(X_train_scaled, y_train)
        test_dataset = SequenceDataset(X_test_scaled, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = FullSequenceModel().to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        train_losses = []
        train_accuracies = []
        
        print(f"\n🚀 Training Full Sequence Model (Target: 76% accuracy)")
        print(f"📊 Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print("=" * 60)
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                data, targets = data.to(self.device), targets.squeeze().to(self.device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_accuracy = 100. * correct / total
            avg_loss = train_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)
            
            # Validation
            if epoch % 5 == 0:
                val_accuracy = self.evaluate_model(model, test_loader)
                scheduler.step(avg_loss)
                
                print(f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_accuracy:.1f}% | Val Acc: {val_accuracy:.1f}%")
        
        # Final evaluation
        final_accuracy = self.evaluate_model(model, test_loader)
        print(f"\n🎯 Final Model Performance:")
        print(f"   Accuracy: {final_accuracy:.1f}% (Target: 76%)")
        print(f"   Status: {'✅ PASSED' if final_accuracy >= 76 else '⚠️  BELOW TARGET'}")
        
        return model, scaler, final_accuracy
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model performance"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in test_loader:
                data, targets = data.to(self.device), targets.squeeze().to(self.device)
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total

def main():
    parser = argparse.ArgumentParser(description='Phase 2: Sequence Model Training')
    parser.add_argument('--data-dir', type=str, default='data/raw/crypto_patterns', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=200, help='Sequence length')
    
    args = parser.parse_args()
    
    print("🚀 Phase 2: Sequence Model Training Framework")
    print(f"🎯 Target: 76% accuracy (research baseline)")
    print(f"📁 Data source: {args.data_dir}")
    print(f"🔧 Sequence length: {args.sequence_length}")
    
    # Initialize processor
    processor = PatternSequenceProcessor(sequence_length=args.sequence_length)
    
    # Load and process data
    print("\n📊 Loading pattern data...")
    sequences, labels = processor.load_pattern_data(args.data_dir)
    
    if len(sequences) == 0:
        print("❌ No pattern data found. Run Phase 1 data collection first.")
        return
    
    print(f"✅ Loaded {len(sequences)} sequences")
    print(f"📈 Label distribution: {np.bincount(labels)}")
    
    # Train model
    trainer = ModelTrainer()
    model, scaler, accuracy = trainer.train_sequence_model(
        sequences, labels, epochs=args.epochs, batch_size=args.batch_size
    )
    
    # Save model
    model_dir = Path('models/phase2')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
        'accuracy': accuracy,
        'sequence_length': args.sequence_length,
        'timestamp': datetime.now().isoformat()
    }, model_dir / 'full_sequence_model.pth')
    
    print(f"\n💾 Model saved to: {model_dir / 'full_sequence_model.pth'}")
    print(f"🎯 Ready for Phase 3: Real-time Pipeline Development")

if __name__ == "__main__":
    main() 