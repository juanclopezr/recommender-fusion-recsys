import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from tqdm import tqdm
import numpy as np


# BERT4Rec Model
class BERT4Rec(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, embedding_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate, padding_idx=0):
        """
        Build a Transformer-based model for sequential recommendation.
        vocab_size: int, size of the vocabulary (number of unique courses + 1 MASK token + 1 for padding)
        max_sequence_length: int, maximum length of input sequences
        embedding_dim: int, dimension of the embedding vectors
        num_heads: int, number of attention heads
        ff_dim: int, dimension of the feed-forward network
        num_transformer_blocks: int, number of transformer blocks
        dropout_rate: float, dropout rate
        """
        super(BERT4Rec, self).__init__()
        
        self.vocab_size = vocab_size # including padding token but not mask token
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Item embedding
        self.item_embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=self.padding_idx)
        
        # Positional embedding
        self.position_embedding = nn.Embedding(max_sequence_length, embedding_dim)
        
        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_blocks)
        
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
   
    def forward(self, input_ids, return_mask=True):
        """
        Forward pass
        input_ids: [batch_size, sequence_length]
        """
        batch_size, seq_length = input_ids.size()
        
        # Create position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)

        # Embeddings
        item_embeddings = self.item_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)

        embeddings = item_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Create attention mask (mask padding tokens)
        attention_mask = (input_ids != self.padding_idx)
        
        # Invert attention mask for nn.TransformerEncoder (True = masked)
        src_key_padding_mask = ~attention_mask
        
        # Transformer
        transformer_output = self.transformer(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Output predictions
        logits = self.output_layer(transformer_output)
        if return_mask:
            return logits, attention_mask  # Para calcular la loss
        else:
            return logits  # En evaluación, solo logits
    
    def get_embeddings(self, input_ids):
        """
        Get the final hidden representations for each position
        Useful for extracting user representations
        """
        batch_size, seq_length = input_ids.size()
        
        # Create position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)
        
        # Embeddings
        item_embeddings = self.item_embedding(input_ids)
        position_embeddings = self.position_embedding(position_ids)
        embeddings = item_embeddings + position_embeddings
        embeddings = self.dropout(embeddings)
        
        # Create attention mask
        attention_mask = (input_ids != self.padding_idx)
        src_key_padding_mask = ~attention_mask
        
        # Transformer
        transformer_output = self.transformer(
            embeddings, 
            src_key_padding_mask=src_key_padding_mask
        )
        
        return transformer_output


# Loss function
def masked_cross_entropy_loss(logits, targets, mask, ignore_index=-100):
    """
    Compute masked cross-entropy loss
    logits: [batch_size, seq_length, vocab_size]
    targets: [batch_size, seq_length]
    mask: [batch_size, seq_length] (1 for valid positions, 0 for padding)
    ignore_index: int, index to ignore in the loss computation
    """
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    mask = mask.view(-1)

    loss = F.cross_entropy(logits, targets, ignore_index=ignore_index, reduction='none')
    loss = (loss * mask.float()).sum() / mask.sum().clamp(min=1)
    return loss



# Accuracy function
def masked_accuracy(logits, targets, mask):
    """
    Compute masked accuracy
    logits: [batch_size, seq_length, vocab_size]
    targets: [batch_size, seq_length]
    mask: [batch_size, seq_length] (1 for valid positions, 0 for padding)
    """
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    mask = mask.view(-1)

    preds = logits.argmax(dim=-1)
    correct = (preds == targets) * mask
    return correct.sum() / mask.sum().clamp(min=1)



# Training function using external dataloader
def train_model(model, train_loader, val_loader, epochs=30, learning_rate=1e-4, 
                checkpoint_filepath=None, device='cuda', verbose=0):
    """
    Train the BERT4Rec model using external dataloaders
    """
    # Move model to device
    model = model.to(device)
    
    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_train_batches = 0
        
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions, mask = model(batch_inputs)
            
            # Compute loss
            loss = masked_cross_entropy_loss(predictions, batch_targets, mask=mask)
            acc = masked_accuracy(predictions, batch_targets, mask=mask)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += acc.item()
            num_train_batches += 1
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                predictions, mask = model(batch_inputs)
                loss = masked_cross_entropy_loss(predictions, batch_targets, mask=mask)
                acc = masked_accuracy(predictions, batch_targets, mask=mask)

                val_loss += loss.item()
                val_acc += acc.item()
                num_val_batches += 1
        
        # Average metrics
        avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0
        avg_train_acc = train_acc / num_train_batches if num_train_batches > 0 else 0
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        avg_val_acc = val_acc / num_val_batches if num_val_batches > 0 else 0
        
        if verbose > 0:
            print(f'Epoch [{epoch+1}/{epochs}] - '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        
        # Save best model
        if checkpoint_filepath and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_filepath)
            if verbose > 0:
                print(f'Model saved to {checkpoint_filepath}')
    
    return model


# Load model weights
def load_model_weights(model, checkpoint_filepath, device='cuda'):
    """
    Load model weights from a checkpoint file
    """
    if os.path.exists(checkpoint_filepath):
        model.load_state_dict(torch.load(checkpoint_filepath, map_location=device))
        print(f"Model weights loaded from {checkpoint_filepath}")
    else:
        print(f"Checkpoint file {checkpoint_filepath} does not exist.")
    return model


# Helper function for padding sequences
def _pad_sequence(seq, max_sequence_length, padding_token=0):
    """Helper function to pad sequences"""
    if len(seq) >= max_sequence_length:
        # Take the last max_sequence_length elements
        return seq[-max_sequence_length:]
    else:
        padding_length = max_sequence_length - len(seq)
        return [padding_token] * padding_length + seq

# Recommendation generation
def generate_recommendations(model, sequence, course_encoder, max_sequence_length, k, mask_token, 
                           padding_token=0, device='cuda'):
    """
    Generate k course recommendations based on a user's course sequence
    """
    model.eval()
    current_sequence = list(sequence)
    recommended_courses_names = []
    recommended_courses_token = []
    
    with torch.no_grad():
        for _ in range(k):
            current_sequence.append(mask_token)
            
            # Prepare input - pad sequence manually
            padded_seq = _pad_sequence(current_sequence, max_sequence_length, padding_token)
            input_seq = torch.tensor(padded_seq, dtype=torch.long).unsqueeze(0).to(device)

            # Predict
            predictions = model(input_seq, return_mask=False) 
            
            # Get prediction for the last position

            last_token_pred = predictions[0, -1, :]  # [vocab_size-1]
            predicted_index = torch.argmax(last_token_pred).item()
            
            # Decode predicted course
            if predicted_index < len(course_encoder.classes_):  
                predicted_course_encoded = predicted_index 
                predicted_course_name = course_encoder.inverse_transform([predicted_course_encoded])[0]
                
                # Add to sequence and results
                current_sequence.append(predicted_course_encoded)
                recommended_courses_names.append(predicted_course_name)
                recommended_courses_token.append(predicted_course_encoded)
            else:
                print(f"Warning: Predicted index {predicted_index} is out of bounds.")
                break
    
    return recommended_courses_names, recommended_courses_token


def generate_recommendations_test_dataset(model, encoded_train_sequences, encoded_test_sequences, 
                                        course_encoder, mask_token, max_sequence_length, k, device='cuda', padding_token=0):
    """
    Evaluate the model on the test set
    """
    users = encoded_test_sequences.index
    courses_recommended_list = []
    courses_test_dataset = []
    num_evaluated_users = 0
    
    print(f"Evaluating model on {len(users)} common users...")
    
    for user_id in tqdm(users):
        train_sequence = encoded_train_sequences.loc[user_id]
        test_sequence = encoded_test_sequences.loc[user_id]
        
        if len(train_sequence) > 0 and len(test_sequence) > 0:
            num_evaluated_users += 1
            current_sequence = list(train_sequence)
            
            # Generate recommendations
            _, recommended_courses_encoded = generate_recommendations(
                model, current_sequence, course_encoder, max_sequence_length, k, 
                mask_token=mask_token, device=device, padding_token=padding_token
            )
            
            courses_recommended_list.append(recommended_courses_encoded)
            courses_test_dataset.append(test_sequence)
    
    print(f"Evaluated {num_evaluated_users} users.")
    return courses_recommended_list, courses_test_dataset


# Factory function for creating the model
def create_model(vocab_size, max_sequence_length=10, embedding_dim=128, 
                         num_heads=8, ff_dim=512, num_transformer_blocks=6, dropout_rate=0.1, padding_idx=0):
    """
    Factory function to create BERT4Rec model
    """
    return BERT4Rec(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        dropout_rate=dropout_rate,
        padding_idx=padding_idx
    )


def save_pytorch_weights(model, save_path):
    """
    Save PyTorch model weights
    
    Args:
        model: PyTorch model
        save_path: Path to save the weights
    """
    torch.save(model.state_dict(), save_path)
    print(f"PyTorch model weights saved to {save_path}")


def load_pytorch_weights(model, weights_path, device='cuda'):
    """
    Load PyTorch model weights
    
    Args:
        model: PyTorch model
        weights_path: Path to the saved weights
        device: Device to load the model on
    
    Returns:
        model: Model with loaded weights
    """
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"PyTorch model weights loaded from {weights_path}")
    else:
        print(f"Weight file {weights_path} does not exist")
    
    return model.to(device)
