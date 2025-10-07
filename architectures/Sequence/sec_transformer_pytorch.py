import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from tqdm import tqdm
import numpy as np


# Sequential Transformer Model
class SequentialTransformer(nn.Module):
    def __init__(self, vocab_size, max_sequence_length, embedding_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate):
        """
        Build a Transformer-based model for course recommendation.
        vocab_size: int, number of unique courses + 1 (for padding)
        max_sequence_length: int, maximum length of input sequences
        embedding_dim: int, dimension of the embedding vectors
        num_heads: int, number of attention heads
        ff_dim: int, dimension of the feed-forward network
        num_transformer_blocks: int, number of transformer blocks
        dropout_rate: float, dropout rate for regularization
        """
        super(SequentialTransformer, self).__init__()
        
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.padding_idx = 0
        
        # Item embedding
        self.item_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=self.padding_idx)
        
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
        
        # Output layer - predict next course (exclude padding token from output)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    
    def forward(self, input_ids):
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
        
        # Extract the representation of the last item in the sequence for prediction
        last_token_representation = transformer_output[:, -1, :]  # [batch_size, embedding_dim]
        
        # Final dense layer for prediction
        logits = self.output_layer(last_token_representation)  # [batch_size, vocab_size-1]
        
        return logits


def cross_entropy_loss(predictions, targets):
    """
    Compute cross-entropy loss for next-item prediction.

    predictions: [batch_size, vocab_size]
    targets: [batch_size, 1] or [batch_size]
    """
    if targets.dim() > 1:
        targets = targets.squeeze(-1)  # -> [batch_size]

    loss = F.cross_entropy(predictions, targets)
    return loss


def accuracy(predictions, targets):
    """
    Compute top-1 accuracy for next-item prediction.

    predictions: [batch_size, vocab_size]
    targets: [batch_size, 1] or [batch_size]
    """
    if targets.dim() > 1:
        targets = targets.squeeze(-1)

    preds = predictions.argmax(dim=-1)  # [batch_size]
    correct = (preds == targets).float()
    return correct.mean()



def train_model(model, train_loader, val_loader, epochs=30, learning_rate=1e-4, 
                checkpoint_filepath=None, device='cuda', verbose=1):

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0

        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            optimizer.zero_grad()

            predictions = model(batch_inputs)  # logits (no softmax)
            loss = cross_entropy_loss(predictions, batch_targets)
            acc = accuracy(predictions, batch_targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += acc.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)

        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                predictions = model(batch_inputs)
                val_loss += cross_entropy_loss(predictions, batch_targets).item()
                val_acc += accuracy(predictions, batch_targets).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)

        if verbose:
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                  f"Train Acc: {avg_train_acc:.4f} | Val Acc: {avg_val_acc:.4f}")

        # Save best model
        if checkpoint_filepath and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_filepath)
            if verbose:
                print(f"✅ Saved best model to {checkpoint_filepath}")

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
        # Use the last max_sequence_length items if longer than max length
        return seq[-max_sequence_length:]
    else:
        padding_length = max_sequence_length - len(seq)
        return [padding_token] * padding_length + seq

# Recommendation generation
def generate_recommendations(model, sequence, course_encoder, max_sequence_length, k, 
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
            # Prepare input - pad sequence manually
            padded_seq = _pad_sequence(current_sequence, max_sequence_length, padding_token)
            input_seq = torch.tensor(padded_seq, dtype=torch.long).unsqueeze(0).to(device)
            
            # Predict
            predictions = model(input_seq)
            
            # Get prediction
            predicted_probs = predictions[0]  # [vocab_size-1]
            predicted_index = torch.argmax(predicted_probs).item()
            
            # Decode predicted course
            if predicted_index < len(course_encoder.classes_):  # Exclude padding token
                predicted_course_encoded = predicted_index  # Adjust for padding token
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
                                        course_encoder, max_sequence_length, k, device='cuda'):
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
                model, current_sequence, course_encoder, max_sequence_length, k, device=device
            )
            
            courses_recommended_list.append(recommended_courses_encoded)
            courses_test_dataset.append(test_sequence)
    
    print(f"Evaluated {num_evaluated_users} users.")
    return courses_recommended_list, courses_test_dataset


# Factory function for creating the model
def create_model(vocab_size, max_sequence_length=50, embedding_dim=128, 
                                      num_heads=8, ff_dim=512, num_transformer_blocks=6, dropout_rate=0.1):
    """
    Factory function to create Sequential Transformer model
    """
    return SequentialTransformer(
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        dropout_rate=dropout_rate
    )


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