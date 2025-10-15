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
                 use_bpr=False, dropout_rate=0.2, layers_per_modality=[512, 256, 128], fusion_layers=[64, 32], autoencoders=None, autoencoder_output_dim=None):
        """
        Args:
            modality_dims (dict): Dictionary mapping the respective 2 vectors (user, course) with the dimension
            Example: {'user': 128, 'course': 128}
            shared_dim (int): Dimension of shared representation space
            fusion_method (str): Method for fusing modalities ('concat', 'by_autoencoder'.)
            use_bpr (bool): Whether to use BPR (Bayesian Personalized Ranking) approach, False for binary classification
            dropout_rate (float): Dropout rate for regularization
            layers_per_modality (list): Number of dense layers per modality
            autoencoders (dict or list): Dictionary or list mapping modality names ('user', 'course') to autoencoder models. Only used if fusion_method == 'by_autoencoder'.
            autoencoder_output_dim (int): Output dimension of the autoencoder encoder. Only used if fusion_method == 'by_autoencoder'.
        """
        super(MultimodalModel, self).__init__()
        self.modality_dims = modality_dims
        self.shared_dim = shared_dim
        self.fusion_method = fusion_method
        self.use_bpr = use_bpr
        self.dropout_rate = dropout_rate
        self.user_vec_dim = modality_dims['user']
        self.course_vec_dim = modality_dims['course']
        self.layers_per_modality = layers_per_modality
        self.fusion_layers = fusion_layers
        self.autoencoder_output_dim = autoencoder_output_dim

        # Store autoencoders for each modality (if provided)
        self.autoencoders = autoencoders if autoencoders is not None else {}

        #Freze autoencoder parameters if using autoencoder fusion
        if self.fusion_method == 'by_autoencoder' and self.autoencoders:
            for _, autoencoder in self.autoencoders.items():
                for param in autoencoder.parameters():
                    param.requires_grad = False

        # Define feature layers for user and course modalities
        if self.fusion_method == 'concat':
            dimensions_user_layer = self.user_vec_dim
        else:
            dimensions_user_layer = self.autoencoder_output_dim

        print(f"User feature layer input dim: {dimensions_user_layer}")
        self.user_feature_layer = self._feature_layer(
            input_dim=dimensions_user_layer,
            output_dim=self.shared_dim,
            number_of_layers=self.layers_per_modality,
            dropout_rate=self.dropout_rate
        )

        if self.fusion_method == 'concat':
            dimensions_course_layer = self.course_vec_dim
        else:
            dimensions_course_layer = self.autoencoder_output_dim

        print(f"Course feature layer input dim: {dimensions_course_layer}")
        self.course_feature_layer = self._feature_layer(
            input_dim=dimensions_course_layer,
            output_dim=self.shared_dim,
            number_of_layers=self.layers_per_modality,
            dropout_rate=self.dropout_rate
        )

        # # Learnable fusion weights for modalities
        # self.alpha_user = nn.Parameter(torch.tensor(0.5))
        # self.alpha_course = nn.Parameter(torch.tensor(0.5))


        if not self.use_bpr:
            # Fusion layer for combined user and course features for binary classification
            self.fusion_layer_binary = self._feature_layer(
                input_dim=self.shared_dim * 2,
                output_dim=1, # Binary classification output
                number_of_layers=self.fusion_layers,
                dropout_rate=self.dropout_rate
            )

    def _apply_autoencoder(self, x, modality):
        """
        Apply the encoder part of the autoencoder to the input x for the given modality.
        """
        if self.fusion_method == 'by_autoencoder' and modality in self.autoencoders:
            return self.autoencoders[modality].encode(x)
        return x
    
    def forward(self, user, course_positive, course_negative=None ):
        if self.fusion_method == 'by_autoencoder':
            user_emb = self._apply_autoencoder(user, 'user')
            course_pos_emb = self._apply_autoencoder(course_positive, 'course')
            course_neg_emb = self._apply_autoencoder(course_negative, 'course') if (self.use_bpr and course_negative is not None) else None
        else:
            user_emb = user
            course_pos_emb = course_positive
            course_neg_emb = course_negative if (self.use_bpr and course_negative is not None) else None

        # Pass through feature layers
        user_feat = self.user_feature_layer(user_emb)
        course_pos_feat = self.course_feature_layer(course_pos_emb)
        course_neg_feat = self.course_feature_layer(course_neg_emb) if course_neg_emb is not None else None

        user_feat = F.normalize(user_feat, p=2, dim=-1)
        course_pos_feat = F.normalize(course_pos_feat, p=2, dim=-1)
        if course_neg_feat is not None:
            course_neg_feat = F.normalize(course_neg_feat, p=2, dim=-1)

        if not self.use_bpr:
            fusion_input = torch.cat([user_feat, course_pos_feat], dim=-1)
            out = self.fusion_layer_binary(fusion_input)
            return out
        else:
            if course_neg_feat is not None:
                return user_feat, course_pos_feat, course_neg_feat
            else:
                similarity_scores = torch.sum(user_feat * course_pos_feat, dim=-1, keepdim=True)
                return similarity_scores

    def _feature_layer(self, input_dim, output_dim, number_of_layers=[512, 256, 128], activation=nn.LeakyReLU(), dropout_rate=0.2):
            """
                Helper function to create a dense layer with activation and dropout.
            Args:
                input_dim (int): Dimension of input features
                output_dim (int): Dimension of output features
                number_of_layers (list): List of dense layer sizes to create
                activation (nn.Module): Activation function to use
                dropout_rate (float): Dropout rate for regularization
            Returns:
                nn.Sequential: Sequential model containing the layers     
            """
            layers = []
            prev_dim = input_dim
            
            for dim in number_of_layers:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(activation)
                layers.append(nn.Dropout(dropout_rate))
                prev_dim = dim
            
            layers.append(nn.Linear(prev_dim, output_dim))
            return nn.Sequential(*layers)
    

    def loss_bpr_function(self, user_feat, course_pos_feat, course_neg_feat, reg_lambda=5e-3):
        """
        BPR Loss Function with dot product computed inside the loss.

        Args:
            user_feat: Tensor (batch_size, embedding_dim) - user representations
            course_pos_feat: Tensor (batch_size, embedding_dim) - positive item representations
            course_neg_feat: Tensor (batch_size, embedding_dim) - negative item representations
            reg_lambda: float, regularization coefficient

        Returns:
            loss: scalar tensor with the BPR loss value
        """
        pos_scores = torch.sum(user_feat * course_pos_feat, dim=1)
        neg_scores = torch.sum(user_feat * course_neg_feat, dim=1)
        diff = pos_scores - neg_scores
        loss = -torch.mean(F.logsigmoid(diff))

        # Regularización L2
        if reg_lambda > 0.0:
            reg_term = (
                user_feat.norm(2).pow(2) +
                course_pos_feat.norm(2).pow(2) +
                course_neg_feat.norm(2).pow(2)
            ) / user_feat.size(0)

            loss = loss + reg_lambda * reg_term

        return loss
    
    def loss_binary_function(self, predictions, targets):
        """
        Compute binary cross-entropy loss given predictions and targets.
        """
        loss = F.binary_cross_entropy_with_logits(predictions.squeeze(), targets.float())
        return loss
    
    def predict_scores(self, user, course_positive):
        """
        Compute similarity scores for user-course pairs during inference.
        This method is specifically for inference mode and doesn't require negative courses.
        
        Args:
            user: User feature vectors
            course_positive: Course feature vectors
            
        Returns:
            Similarity scores between users and courses
        """
        self.eval()
        with torch.no_grad():
            # Call forward with course_negative=None to trigger inference mode
            scores = self.forward(user, course_positive, course_negative=None)
            return scores
    
    def train_model(self, train_loader, val_loader=None, epochs=10, lr=1e-3, save_path=None, device='cuda', reg_lambda=5e-3, verbose=True, early_stopping_patience=5, early_stopping_delta=1e-5):
        """
        Train the multimodal model with validation support, best model saving, and early stopping.
        Args:
            train_loader (DataLoader): DataLoader providing training data
            val_loader (DataLoader, optional): DataLoader providing validation data
            epochs (int): Number of training epochs
            lr (float): Learning rate for optimizer
            save_path (str, optional): Path to save the best model
            device (str): Device to train on ('cpu' or 'cuda')
            reg_lambda (float): Regularization lambda for BPR loss
            verbose (bool): Whether to print training progress
            early_stopping_patience (int): Number of epochs to wait for improvement before stopping (default: 10)
            early_stopping_delta (float): Minimum change in loss to qualify as an improvement (default: 1e-6)
        """
        # Move model to device
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        
        # Initialize tracking variables
        best_val_loss = float('inf')
        best_model_state = None
        train_losses = []
        val_losses = []
        
        # Early stopping variables
        early_stopping_counter = 0
        early_stopped = False

        if verbose:
            print(f"Training multimodal model for {epochs} epochs...")
            if val_loader is not None:
                print(f"Using validation dataset with {len(val_loader)} batches")
                print(f"Early stopping: patience={early_stopping_patience}, delta={early_stopping_delta}")
            else:
                print("No validation dataset provided - using training loss for model selection")
                print(f"Early stopping: patience={early_stopping_patience}, delta={early_stopping_delta}")

        for epoch in range(epochs):
            # Training phase
            self.train()
            total_train_loss = 0
            num_train_batches = 0
            
            for batch in train_loader:
                # Move batch data to device
                user = batch['user'].to(device)
                course_positive = batch['course_positive'].to(device)
                if self.use_bpr:
                    course_negative = batch['course_negative'].to(device)
                else:
                    course_negative = None
                targets = batch.get('targets', None)
                if targets is not None:
                    targets = targets.to(device)
                
                optimizer.zero_grad()
                
                if self.use_bpr:
                    user_feat, course_pos_feat, course_neg_feat = self.forward(user, course_positive, course_negative)
                    loss = self.loss_bpr_function(user_feat, course_pos_feat, course_neg_feat, reg_lambda=reg_lambda)
                else:
                    predictions = self.forward(user, course_positive)
                    loss = self.loss_binary_function(predictions, targets)
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                num_train_batches += 1
            
            avg_train_loss = total_train_loss / num_train_batches
            train_losses.append(avg_train_loss)
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                total_val_loss = 0
                num_val_batches = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        # Move batch data to device
                        user = batch['user'].to(device)
                        course_positive = batch['course_positive'].to(device)
                        if self.use_bpr:
                            course_negative = batch['course_negative'].to(device)
                        else:
                            course_negative = None
                        targets = batch.get('targets', None)
                        if targets is not None:
                            targets = targets.to(device)
                        
                        if self.use_bpr:
                            user_feat, course_pos_feat, course_neg_feat = self.forward(user, course_positive, course_negative)
                            val_loss = self.loss_bpr_function(user_feat, course_pos_feat, course_neg_feat, reg_lambda=reg_lambda)
                        else:
                            predictions = self.forward(user, course_positive)
                            val_loss = self.loss_binary_function(predictions, targets)
                        
                        total_val_loss += val_loss.item()
                        num_val_batches += 1
                
                avg_val_loss = total_val_loss / num_val_batches
                val_losses.append(avg_val_loss)
                current_val_loss = avg_val_loss
            else:
                # If no validation data, use training loss for model selection
                current_val_loss = avg_train_loss
                val_losses.append(avg_train_loss)
            
            # Save best model based on validation loss (or training loss if no validation)
            if current_val_loss < best_val_loss - early_stopping_delta:
                best_val_loss = current_val_loss
                best_model_state = self.state_dict().copy()
                early_stopping_counter = 0  # Reset counter on improvement
                if save_path:
                    torch.save(best_model_state, save_path)
                    if verbose:
                        print(f" New best model saved at epoch {epoch+1} with val_loss: {best_val_loss:.6f}")
            else:
                # No improvement
                early_stopping_counter += 1
                if verbose and early_stopping_counter > 0:
                    print(f" No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")
            
            # Check early stopping condition
            if early_stopping_counter >= early_stopping_patience:
                early_stopped = True
                if verbose:
                    print(f" Early stopping triggered after {epoch+1} epochs (patience: {early_stopping_patience})")
                break

            # Print progress
            if val_loader is not None and verbose:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {avg_val_loss:.6f}, Best Val Loss: {best_val_loss:.6f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.6f}, "
                        f"Best Loss: {best_val_loss:.6f}")
        # Load best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)
            if verbose:
                completion_msg = " Training completed"
                if early_stopped:
                    completion_msg += f" (early stopped at epoch {epoch+1}/{epochs})"
                else:
                    completion_msg += f" (full {epochs} epochs)"
                
                if val_loader is not None:
                    print(f"{completion_msg}. Best validation loss: {best_val_loss:.6f}")
                else:
                    print(f"{completion_msg}. Best training loss: {best_val_loss:.6f}")


    def generate_k_recommendations(self, course_embeddings_dict, user_tensor, courses_already_taken, k=5, device='cuda'):
        """
        Generate top-k course recommendations for a user based on similarity scores.
        
        Args:
            course_embeddings_dict (dict): Dictionary mapping course_id to course embedding tensors
                                         Example: {'course_1': tensor, 'course_2': tensor, ...}
            user_tensor (torch.Tensor): User embedding tensor of shape [1, user_dim] or [num_users, user_dim]
            k (int): Number of top recommendations to return (default: 5)
            device (str): Device to run inference on ('cpu' or 'cuda')
            
        Returns:
            list: List of course_id sorted by score in descending order
        """
        self.to(device)
        self.eval()

        if user_tensor.dim() == 1:
            user_tensor = user_tensor.unsqueeze(0)
        user_tensor = user_tensor.to(device)

        course_ids = list(course_embeddings_dict.keys())
        course_embs = torch.stack([course_embeddings_dict[cid] for cid in course_ids]).to(device)

        # Compute similarity scores between user and all courses
        with torch.no_grad():
            if self.use_bpr:
                scores = self.predict_scores(
                    user_tensor.unsqueeze(1),     
                    course_embs.unsqueeze(0)       
                ).squeeze(-1)                       
            else:
                scores = []
                for u in user_tensor:
                    # Expand user [dim] to match courses
                    u_expand = u.unsqueeze(0).repeat(course_embs.size(0), 1)
                    s = self.predict_scores(u_expand, course_embs).squeeze()
                    scores.append(s)
                scores = torch.stack(scores)  # [num_users, num_courses]
        # Filter out courses already taken by the user
        for user_idx, taken in enumerate(courses_already_taken):
            taken_set = set(taken)
            for i, cid in enumerate(course_ids):
                if cid in taken_set:
                    scores[user_idx, i] = -float('inf')  # Assign very low score to already taken courses

        # Top-K
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)
        topk_course_ids = [[course_ids[i] for i in idx_row] for idx_row in topk_indices.tolist()]

        return topk_course_ids

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
    def train_autoencoder(autoencoder, data, epochs=20, lr=1e-3, save_path=None, device='cpu', validation_split=0.2, verbose=True):
        """
        Train the autoencoder and save the best model based on lowest validation loss.
        
        Args:
            autoencoder: The autoencoder model to train
            data: Training data tensor
            epochs: Number of training epochs
            lr: Learning rate
            save_path: Path to save the best model (optional)
            device: Device to train on ('cpu' or 'cuda')
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
        
        Returns:
            Trained autoencoder model
        """
        autoencoder = autoencoder.to(device)
        data = data.to(device)
        
        # Split data into training and validation sets
        if validation_split > 0.0:
            num_samples = data.shape[0]
            num_val = int(num_samples * validation_split)
            
            # Shuffle indices for random split
            indices = torch.randperm(num_samples)
            val_indices = indices[:num_val]
            train_indices = indices[num_val:]
            
            train_data = data[train_indices]
            val_data = data[val_indices]
            if verbose:
                print(f"Data split: {len(train_data)} training samples, {len(val_data)} validation samples")
        else:
            train_data = data
            val_data = None
            if verbose:
                print(f"No validation split - using all {len(train_data)} samples for training")
        
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        best_model_state = None
        
        print(f"Training autoencoder for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training phase
            autoencoder.train()
            optimizer.zero_grad()
            train_reconstructed, _ = autoencoder(train_data)
            train_loss = criterion(train_reconstructed, train_data)
            train_loss.backward()
            optimizer.step()
            
            # Validation phase
            autoencoder.eval()
            with torch.no_grad():
                if val_data is not None:
                    val_reconstructed, _ = autoencoder(val_data)
                    val_loss = criterion(val_reconstructed, val_data)
                    current_val_loss = val_loss.item()
                else:
                    # If no validation data, use training loss for model selection
                    current_val_loss = train_loss.item()
            
            current_train_loss = train_loss.item()
            
            # Save best model based on validation loss (or training loss if no validation)
            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                best_model_state = autoencoder.state_dict().copy()
                if save_path:
                    torch.save(best_model_state, save_path)
                    print(f" New best model saved at epoch {epoch+1} with val_loss: {best_val_loss:.6f}")
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:  # Print every 5 epochs and first epoch
                if val_data is not None and verbose:
                    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {current_train_loss:.6f}, "
                          f"Val Loss: {current_val_loss:.6f}, Best Val Loss: {best_val_loss:.6f}")
                else:
                    if verbose:
                        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {current_train_loss:.6f}, "
                          f"Best Loss: {best_val_loss:.6f}")
        
        # Load best model state
        if best_model_state is not None:
            autoencoder.load_state_dict(best_model_state)
            if val_data is not None:
                print(f" Training completed. Best validation loss: {best_val_loss:.6f}")
            else:
                print(f" Training completed. Best training loss: {best_val_loss:.6f}")
        
        return autoencoder
    
            
