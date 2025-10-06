import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Autoencoder(nn.Module):
    """
    Basic Autoencoder for feature reconstruction and dimensionality reduction.
    """
    def __init__(self, input_dim, encoding_dims):
        """
        Args:
            input_dim (int): Dimension of input features
            encoding_dims (list): List of hidden layer dimensions for encoder
                                 Example: [512, 256, 128] for 3-layer encoder
        """
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.encoding_dims = encoding_dims
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Build decoder (symmetric to encoder)
        self.decoder = self._build_decoder()
        
    
    def _build_encoder(self):
        """Build the encoder network"""
        layers = []
        prev_dim = self.input_dim
        
        for dim in self.encoding_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        # Remove last dropout
        if layers:
            layers.pop()
        
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        """Build the decoder network (symmetric to encoder)"""
        layers = []
        
        # Reverse the encoding dimensions
        decoder_dims = self.encoding_dims[::-1][1:] + [self.input_dim]
        prev_dim = self.encoding_dims[-1]  # Start from bottleneck
        
        for i, dim in enumerate(decoder_dims):
            layers.append(nn.Linear(prev_dim, dim))
            if i < len(decoder_dims) - 1:  # Don't add activation after last layer
                layers.extend([nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = dim
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation back to input space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass through autoencoder"""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded, encoded


def create_autoencoder(autoencoder_type='basic', **kwargs):
    """
    Factory function to create different types of autoencoders.
    
    Args:
        autoencoder_type (str): Type of autoencoder ('basic', 'multimodal')
        **kwargs: Additional arguments for specific autoencoder types
    
    Returns:
        nn.Module: Autoencoder model
    """
    if autoencoder_type == 'basic':
        return Autoencoder(
            input_dim=kwargs.get('input_dim', 784),
            encoding_dims=kwargs.get('encoding_dims', [512, 256, 128])
        )
    elif autoencoder_type == 'multimodal':
        pass
    else:
        raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")
