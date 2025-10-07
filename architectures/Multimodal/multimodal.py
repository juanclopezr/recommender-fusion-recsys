import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MultimodalModel(nn.Module):
    """
    Multimodal Model that uses multiple input modalities. Incorporates pre-computed embeddings 
    from other models (BERT, SciBERT, etc.) for both users and courses, focusing on fusion 
    and training with binary classification or BPR approach.
    """
    def __init__(self, modality_dims, shared_dim=128, fusion_method='concat', 
                 use_bpr=False, dropout_rate=0.2):
        """
        Args:
            modality_dims (dict): Dictionary mapping modality names to their embedding dimensions
                Example: {
                    'text': 768,           # BERT embeddings
                    'scibert': 768,        # SciBERT embeddings  
                    'numerical': 50,       # Other features
                    'graph': 128           # Graph embeddings
                }
            shared_dim (int): Dimension of shared representation space
            fusion_method (str): Method for fusing modalities ('concat', 'sum', 'attention', 'mlp')
            use_bpr (bool): Whether to use BPR (Bayesian Personalized Ranking) approach
            dropout_rate (float): Dropout rate for regularization
        """
        super(MultimodalModel, self).__init__()
        self.modality_dims = modality_dims
        self.shared_dim = shared_dim
        self.fusion_method = fusion_method
        self.use_bpr = use_bpr
        self.dropout_rate = dropout_rate
        self.modalities = list(modality_dims.keys())
        
        # Projection layers for each modality to shared space
        self.user_projectors = nn.ModuleDict()
        self.course_projectors = nn.ModuleDict()
        
        for modality, input_dim in modality_dims.items():
            # Project pre-computed embeddings to shared space
            self.user_projectors[modality] = nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim, shared_dim)
            )
            self.course_projectors[modality] = nn.Sequential(
                nn.Linear(input_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim, shared_dim)
            )
        
        # Fusion mechanism
        if fusion_method == 'attention':
            self.user_attention = nn.Parameter(torch.ones(len(self.modalities)))
            self.course_attention = nn.Parameter(torch.ones(len(self.modalities)))
        elif fusion_method == 'mlp':
            fused_dim = shared_dim * len(self.modalities)
            self.user_fusion_mlp = nn.Sequential(
                nn.Linear(fused_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim, shared_dim)
            )
            self.course_fusion_mlp = nn.Sequential(
                nn.Linear(fused_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim, shared_dim)
            )
        
        # Final prediction layers
        final_input_dim = shared_dim * 2  # User + Course representations
        
        if use_bpr:
            # For BPR: output raw scores (no sigmoid)
            self.prediction_layer = nn.Sequential(
                nn.Linear(final_input_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim // 2, 1)
            )
        else:
            # For binary classification: output probabilities
            self.prediction_layer = nn.Sequential(
                nn.Linear(final_input_dim, shared_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim, shared_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(shared_dim // 2, 1),
                nn.Sigmoid()
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def encode_user_modality(self, user_embeddings, modality):
        """
        Project pre-computed user embeddings for a specific modality to shared space
        
        Args:
            user_embeddings (torch.Tensor): Pre-computed embeddings [batch_size, modality_dim]
            modality (str): Name of the modality
        
        Returns:
            torch.Tensor: Projected embeddings in shared space
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Project to shared space
        projected = self.user_projectors[modality](user_embeddings)
        return projected
    
    def encode_course_modality(self, course_embeddings, modality):
        """
        Project pre-computed course embeddings for a specific modality to shared space
        
        Args:
            course_embeddings (torch.Tensor): Pre-computed embeddings [batch_size, modality_dim]
            modality (str): Name of the modality
        
        Returns:
            torch.Tensor: Projected embeddings in shared space
        """
        if modality not in self.modalities:
            raise ValueError(f"Unknown modality: {modality}")
        
        # Project to shared space
        projected = self.course_projectors[modality](course_embeddings)
        return projected
    
    def fuse_modalities(self, modal_representations, entity_type='user'):
        """Fuse multiple modal representations"""
        if self.fusion_method == 'concat':
            return torch.cat(list(modal_representations.values()), dim=-1)
        elif self.fusion_method == 'sum':
            return sum(modal_representations.values())
        elif self.fusion_method == 'attention':
            # Use attention weights
            attention_weights = self.user_attention if entity_type == 'user' else self.course_attention
            attention_weights = F.softmax(attention_weights, dim=0)
            
            weighted_sum = torch.zeros_like(list(modal_representations.values())[0])
            for i, (modality, representation) in enumerate(modal_representations.items()):
                weighted_sum += attention_weights[i] * representation
            
            return weighted_sum
        elif self.fusion_method == 'mlp':
            # Concatenate and pass through MLP
            concatenated = torch.cat(list(modal_representations.values()), dim=-1)
            fusion_mlp = self.user_fusion_mlp if entity_type == 'user' else self.course_fusion_mlp
            return fusion_mlp(concatenated)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
    
    def forward(self, user_embeddings, course_embeddings, negative_course_embeddings=None):
        """
        Forward pass through multimodal model
        
        Args:
            user_embeddings (dict): Pre-computed user embeddings for each modality
                Example: {
                    'text': tensor([batch_size, 768]),
                    'scibert': tensor([batch_size, 768]),
                    'numerical': tensor([batch_size, 50])
                }
            course_embeddings (dict): Pre-computed course embeddings for each modality
            negative_course_embeddings (dict): Pre-computed negative course embeddings for BPR
        
        Returns:
            dict: Model outputs including predictions and representations
        """
        user_modal_representations = {}
        course_modal_representations = {}
        
        # Process each modality
        for modality in self.modalities:
            if modality in user_embeddings:
                user_modal_representations[modality] = self.encode_user_modality(
                    user_embeddings[modality], modality
                )
            
            if modality in course_embeddings:
                course_modal_representations[modality] = self.encode_course_modality(
                    course_embeddings[modality], modality
                )
        
        # Fuse modalities
        user_representation = self.fuse_modalities(user_modal_representations, 'user')
        course_representation = self.fuse_modalities(course_modal_representations, 'course')
        
        # Compute final prediction
        if self.use_bpr and negative_course_embeddings is not None:
            # BPR approach: compute positive and negative scores
            positive_input = torch.cat([user_representation, course_representation], dim=-1)
            positive_score = self.prediction_layer(positive_input).squeeze()
            
            # Encode negative courses
            negative_course_modal_representations = {}
            for modality in self.modalities:
                if modality in negative_course_embeddings:
                    negative_course_modal_representations[modality] = self.encode_course_modality(
                        negative_course_embeddings[modality], modality
                    )
            
            negative_course_representation = self.fuse_modalities(
                negative_course_modal_representations, 'course'
            )
            negative_input = torch.cat([user_representation, negative_course_representation], dim=-1)
            negative_score = self.prediction_layer(negative_input).squeeze()
            
            return {
                'positive_score': positive_score,
                'negative_score': negative_score,
                'user_representation': user_representation,
                'positive_course_representation': course_representation,
                'negative_course_representation': negative_course_representation,
                'user_modal_representations': user_modal_representations,
                'course_modal_representations': course_modal_representations
            }
        else:
            # Binary classification approach
            combined_input = torch.cat([user_representation, course_representation], dim=-1)
            prediction = self.prediction_layer(combined_input).squeeze()
            
            return {
                'prediction': prediction,
                'user_representation': user_representation,
                'course_representation': course_representation,
                'user_modal_representations': user_modal_representations,
                'course_modal_representations': course_modal_representations
            }


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


def create_multimodal_model(modality_dims, **kwargs):
    """
    Factory function to create multimodal recommendation model.
    
    Args:
        modality_dims (dict): Dictionary mapping modality names to embedding dimensions
            Example: {'text': 768, 'scibert': 768, 'numerical': 50}
        **kwargs: Additional arguments
    
    Returns:
        MultimodalModel: Configured multimodal model
    """
    return MultimodalModel(
        modality_dims=modality_dims,
        shared_dim=kwargs.get('shared_dim', 128),
        fusion_method=kwargs.get('fusion_method', 'concat'),
        use_bpr=kwargs.get('use_bpr', False),
        dropout_rate=kwargs.get('dropout_rate', 0.2)
    )


def create_autoencoder(autoencoder_type='basic', **kwargs):
    """
    Factory function to create different types of autoencoders.
    
    Args:
        autoencoder_type (str): Type of autoencoder ('basic')
        **kwargs: Additional arguments for specific autoencoder types
    
    Returns:
        nn.Module: Autoencoder model
    """
    if autoencoder_type == 'basic':
        return Autoencoder(
            input_dim=kwargs.get('input_dim', 784),
            encoding_dims=kwargs.get('encoding_dims', [512, 256, 128])
        )
    else:
        raise ValueError(f"Unknown autoencoder type: {autoencoder_type}")


def bpr_loss(positive_scores, negative_scores):
    """
    Compute BPR (Bayesian Personalized Ranking) loss.
    
    Args:
        positive_scores (torch.Tensor): Scores for positive items
        negative_scores (torch.Tensor): Scores for negative items
    
    Returns:
        torch.Tensor: BPR loss
    """
    return -torch.mean(torch.log(torch.sigmoid(positive_scores - negative_scores)))


def multimodal_loss_function(outputs, targets, loss_type='mse', modal_weights=None):
    """
    Compute loss for multimodal recommendation model.
    
    Args:
        outputs: Model outputs
        targets: Target values
        loss_type (str): Type of loss ('mse', 'bce', 'bpr')
        modal_weights: Weights for different modalities
    
    Returns:
        torch.Tensor: Total loss
    """
    if loss_type == 'bpr':
        return bpr_loss(outputs['positive_score'], outputs['negative_score'])
    elif loss_type == 'mse':
        return F.mse_loss(outputs['prediction'], targets)
    elif loss_type == 'bce':
        return F.binary_cross_entropy(outputs['prediction'], targets)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
