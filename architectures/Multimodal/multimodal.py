import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


class MultimodalModel(nn.Module):
    """
    Multimodal Model that uses multiple input modalities. Incorporates pre-computed embeddings 
    from other models (BERT, SciBERT, etc.) for both users and courses, focusing on fusion 
    and training with binary classification or BPR approach.
    """
    def __init__(self, modality_dims, shared_dim=128, fusion_method='concat', 
                 use_bpr=False, dropout_rate=0.2, layers_per_modality=2, autoencoders=None):
        """
        Args:
            modality_dims (dict): Dictionary mapping the respective 3 vectors (user, course_positive,
            course_negative) with the dimension
            Example: {'user': 128, 'course_positive': 128, 'course_negative': 128}
            shared_dim (int): Dimension of shared representation space
            fusion_method (str): Method for fusing modalities ('concat', 'by_autoencoder'.)
            use_bpr (bool): Whether to use BPR (Bayesian Personalized Ranking) approach, False for binary classification
            dropout_rate (float): Dropout rate for regularization
            layers_per_modality (int): Number of dense layers per modality 
            autoencoders (dict or list): Dictionary or list mapping modality names ('user', 'course_positive', 'course_negative') to autoencoder models. Only used if fusion_method == 'by_autoencoder'.
        """
        super(MultimodalModel, self).__init__()
        self.modality_dims = modality_dims
        self.shared_dim = shared_dim
        self.fusion_method = fusion_method
        self.use_bpr = use_bpr
        self.dropout_rate = dropout_rate
        self.user_vec_dim = modality_dims['user']
        self.course_pos_vec_dim = modality_dims['course_positive']
        # Negative course vector dimension in the case of BPR, otherwise not included
        if use_bpr:
            self.course_neg_vec_dim = modality_dims['course_negative']
        else:
            self.course_neg_vec_dim = 0

        # Store autoencoders for each modality (if provided)
        self.autoencoders = autoencoders if autoencoders is not None else {}

        # Define feature layers for user and course modalities
        self.user_feature_layer = self._feature_layer(
            input_dim=self.user_vec_dim,
            output_dim=self.shared_dim,
            number_of_layers=layers_per_modality,
            activation=nn.ReLU(),
            dropout_rate=dropout_rate,
            reduction_factor=2
        )

        self.course_feature_layer = self._feature_layer(
            input_dim=self.course_pos_vec_dim,
            output_dim=self.shared_dim,
            number_of_layers=layers_per_modality,
            activation=nn.ReLU(),
            dropout_rate=dropout_rate,
            reduction_factor=2
        )

        if self.use_bpr:
            self.course_neg_feature_layer = self._feature_layer(
                input_dim=self.course_neg_vec_dim,
                output_dim=self.shared_dim,
                number_of_layers=layers_per_modality,
                activation=nn.ReLU(),
                dropout_rate=dropout_rate,
                reduction_factor=2
            )

        # Fusion layer for combined user and course features for binary classification
        self.fusion_layer_binary = self._feature_layer(
            input_dim=self.shared_dim * 2,
            output_dim=1, # Binary classification output
            number_of_layers=self.layers_per_modality,
            activation=nn.ReLU(),
            dropout_rate=self.dropout_rate,
            reduction_factor=2
        )

    def _apply_autoencoder(self, x, modality):
        """
        Apply the encoder part of the autoencoder to the input x for the given modality.
        """
        if self.fusion_method == 'by_autoencoder' and modality in self.autoencoders:
            return self.autoencoders[modality].encode(x)
        return x
    
    def forward(self, user, course_positive, course_negative=None):
        # Apply autoencoder encoder if fusion_method is 'by_autoencoder'

        if self.fusion_method == 'by_autoencoder' and not self.autoencoders:
            raise ValueError("Autoencoders must be provided when fusion_method is 'by_autoencoder'.")

        if self.fusion_method == 'by_autoencoder':
            user_emb = self._apply_autoencoder(user, 'user')
            course_pos_emb = self._apply_autoencoder(course_positive, 'course_positive')
            if self.use_bpr and course_negative is not None:
                course_neg_emb = self._apply_autoencoder(course_negative, 'course_negative')
        else:
            user_emb = user
            course_pos_emb = course_positive
            if self.use_bpr and course_negative is not None:
                course_neg_emb = course_negative
            else:
                course_neg_emb = None

        # Pass through feature layers
        user_feat = self.user_feature_layer(user_emb)
        course_pos_feat = self.course_feature_layer(course_pos_emb)
        if self.use_bpr and course_neg_emb is not None:
            course_neg_feat = self.course_neg_feature_layer(course_neg_emb)
        else:
            course_neg_feat = None

        # Example fusion (binary classification)
        if not self.use_bpr:
            fusion_input = torch.cat([user_feat, course_pos_feat], dim=-1)
            out = self.fusion_layer_binary(fusion_input)
            return out
        else:
            # For BPR, return all three features for custom loss
            return user_feat, course_pos_feat, course_neg_feat
        
    def _feature_layer(input_dim, output_dim, number_of_layers=2, activation=nn.ReLU(), dropout_rate=0.2, reduction_factor=2):
            """
                Helper function to create a dense layer with activation and dropout.
                the input dimension is reduced by reduction_factor after each layer.
            Args:
                input_dim (int): Dimension of input features
                output_dim (int): Dimension of output features
                number_of_layers (int): Number of dense layers to create
                activation (nn.Module): Activation function to use
                dropout_rate (float): Dropout rate for regularization
                reduction_factor (int): Factor by which to reduce the dimension after each layer
            Returns:
                nn.Sequential: Sequential model containing the layers     
            """
            layers = []
            prev_dim = input_dim
            for i in range(number_of_layers):
                next_dim = max(output_dim, prev_dim // reduction_factor)
                layers.append(nn.Linear(prev_dim, next_dim))
                if i < number_of_layers - 1:  # Don't add activation after last layer
                    layers.append(activation)
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = next_dim
            return nn.Sequential(*layers)
    
    def loss_bpr_function(self, user_feat, course_pos_feat, course_neg_feat):
        """
        Compute the BPR loss given user features, positive course features, and negative course features.
        """
        pos_scores = torch.sum(user_feat * course_pos_feat, dim=-1)
        neg_scores = torch.sum(user_feat * course_neg_feat, dim=-1)
        loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        return loss
    
    def loss_binary_function(self, predictions, targets):
        """
        Compute binary cross-entropy loss given predictions and targets.
        """
        loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), targets.float())
        return loss
    
    def train_model(self, data_loader, epochs=10, lr=1e-3):
        """
        Train the multimodal model.
        Args:
            data_loader (DataLoader): DataLoader providing training data
            epochs (int): Number of training epochs
            lr (float): Learning rate for optimizer
        """
        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in data_loader:
                user = batch['user']
                course_positive = batch['course_positive']
                if self.use_bpr:
                    course_negative = batch['course_negative']
                else:
                    course_negative = None
                targets = batch.get('targets', None)
                
                optimizer.zero_grad()
                
                if self.use_bpr:
                    user_feat, course_pos_feat, course_neg_feat = self.forward(user, course_positive, course_negative)
                    loss = self.loss_bpr_function(user_feat, course_pos_feat, course_neg_feat)
                else:
                    predictions = self.forward(user, course_positive)
                    loss = self.loss_binary_function(predictions, targets)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")



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

    def loss_function(self, recon_x, x):
        """Compute reconstruction loss (MSE)"""
        loss = F.mse_loss(recon_x, x)
        return loss

    @staticmethod
    def train_autoencoder(autoencoder, data, epochs=20, lr=1e-3):
        autoencoder.train()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        return autoencoder
            
