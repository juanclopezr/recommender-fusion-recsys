import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, LayerNormalization, Dropout, MultiHeadAttention, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
import random
from tqdm import tqdm

# Data preprocessing

def preprocess_data(df):
    """
        Preprocess the data to create user-course interaction sequences.
        df: pd.DataFrame with columns ['user_id', 'course_name', 'timestamp']
        Returns: pd.Series where index is user_id and values are lists of course_names sorted by timestamp
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values(by=['user_id', 'timestamp'])
    user_sequences = df_sorted.groupby('user_id')['course_name'].apply(list)
    return user_sequences

# Encoding

def encode_courses(train_sequences, test_sequences, list_of_courses=None):
    """
        Encode course names to integer IDs
        train_sequences: pd.Series of lists (training data)
        test_sequences: pd.Series of lists (testing data)
        list_of_courses: list of all unique course names (optional)
        Returns: encoded_train_sequences, encoded_test_sequences, course_encoder
    """
    if list_of_courses is None:
        all_course_names = np.concatenate([np.concatenate(train_sequences.values), np.concatenate(test_sequences.values)])
        unique_course_names = np.unique(all_course_names)
    else:
        unique_course_names = np.array(list_of_courses)

    course_encoder = LabelEncoder()
    course_encoder.fit(unique_course_names)

    #Change to 1-based without including PADDING index 0
    course_encoder.classes_ = np.insert(course_encoder.classes_, 0, "<PAD>")


    try:
        encoded_train_sequences = train_sequences.apply(lambda x: (course_encoder.transform(x)).tolist())
        encoded_test_sequences = test_sequences.apply(lambda x: (course_encoder.transform(x)).tolist())
        return encoded_train_sequences, encoded_test_sequences, course_encoder
    except Exception as e:
        print(f"Error occurred while encoding courses: {e} check if all courses in test set are in training set or if list_of_courses includes all courses.")
        return pd.Series(dtype=int), pd.Series(dtype=int), course_encoder

# Sequence creation

def create_sequences_inputs(sequences, max_sequence_length, mask_token, mask_prob=0.2, mode="MASKED", min_sequence_length=4):
    """
        Create input and target sequences with masking for BERT4Rec.
        sequences: list of lists, where each sublist is a sequence of course IDs
        max_sequence_length: int, maximum length of sequences
        padding_token: int, token used for padding
        mask_token: int, token used for masking
        mask_prob: float, probability of masking a token
        min_sequence_length: int, minimum length for the sub-sequences when using MIXED mode
        mode: str, 
            "MASKED" for BERT4Rec style masking
            "MIXED" for a mix of masking and augmentation strategies using sub-sequences keeping the first elements and splitting longer sequences into shorter ones
        Returns: 
            inputs and targets as numpy arrays
    """

    expanded_sequences = []

    if mode == "MIXED":
        for seq in sequences:
            if len(seq) > min_sequence_length:
                for length in range((len(seq)-min_sequence_length)):
                    sub_seq = seq[0:length+min_sequence_length]
                    expanded_sequences.append(sub_seq)
            else:
                expanded_sequences.append(seq)
        # Shuffle the expanded sequences to ensure randomness
        random.shuffle(expanded_sequences)
        sequences = expanded_sequences


    inputs, targets = [], []
    for seq in sequences:
        initial_size = min(len(seq), max_sequence_length)
        seq = seq[:initial_size]
        masked_input = list(seq)
        # index 0 is reserved for padding
        target = [0] * initial_size
        masked_count_tokens = 0
        for i in range(initial_size):
            if random.random() < mask_prob and masked_count_tokens < initial_size:
                masked_count_tokens += 1
                target[i] = masked_input[i]
                masked_input[i] = mask_token
            if i == initial_size - 1 and masked_count_tokens == 0:
                target[i] = masked_input[i]
                masked_input[i] = mask_token
        inputs.append(masked_input)
        targets.append(target)
    return inputs, targets

# Model building

def build_transformer_model(vocab_size, max_sequence_length, embedding_dim, num_heads, ff_dim, num_transformer_blocks, dropout_rate):
    """
        Build a Transformer-based model for sequential recommendation.
        vocab_size: int, size of the vocabulary (number of unique courses + 1 MasK 
            token + 1 for padding (Paddinng token is included in encode_courses function))
        max_sequence_length: int, maximum length of input sequences
        embedding_dim: int, dimension of the embedding vectors
        num_heads: int, number of attention heads
        ff_dim: int, dimension of the feed-forward network
        num_transformer_blocks: int, number of transformer blocks
        dropout_rate: float, dropout rate
        Returns: compiled Keras Model
    """
    inputs = Input(shape=(max_sequence_length,), dtype=tf.int32)
    item_embeddings = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(inputs)
    position_embeddings = Embedding(input_dim=max_sequence_length, output_dim=embedding_dim)(tf.range(start=0, limit=max_sequence_length, delta=1))
    embeddings = item_embeddings + position_embeddings
    x = embeddings
    for _ in range(num_transformer_blocks):
        attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim // num_heads)(x, x)
        attention_output = Dropout(dropout_rate)(attention_output)
        x = LayerNormalization(epsilon=1e-6)(x + attention_output)
        ffn_output = Dense(ff_dim, activation="relu")(x)
        ffn_output = Dense(embedding_dim)(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)
        x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
    outputs = Dense(vocab_size-1, activation='softmax')(x)
    return Model(inputs, outputs)

# Custom loss and accuracy

def masked_sparse_categorical_crossentropy(y_true, y_pred, padding_token=0):
    mask = tf.not_equal(y_true, padding_token)
    loss = SparseCategoricalCrossentropy(from_logits=False, reduction='none')(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask))

def masked_sparse_categorical_accuracy(y_true, y_pred, padding_token=0):
    mask = tf.not_equal(y_true, padding_token)
    accuracy = tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
    mask = tf.cast(mask, dtype=accuracy.dtype)
    accuracy *= mask
    return tf.reduce_sum(accuracy) / (tf.reduce_sum(mask))

# Training pipeline

def train_model(train_inputs, train_targets, val_inputs, val_targets, model, sequence_size, epochs=30, checkpoint_filepath=None, padding_token=0, verbose=0):
    """
        Train the Transformer-based model.
        train_inputs: np.array, training input sequences
        train_targets: np.array, training target sequences
        val_inputs: np.array, validation input sequences
        val_targets: np.array, validation target sequences
        model: Keras Model
        padding_token: int, token used for padding (default is 0)
        epochs: int, number of training epochs
        checkpoint_filepath: str or None, path to save model checkpoints
        Returns: trained Keras Model 
    """
    model.compile(optimizer=Adam(),
                  loss=lambda y_true, y_pred: masked_sparse_categorical_crossentropy(y_true, y_pred, padding_token),
                  metrics=[lambda y_true, y_pred: masked_sparse_categorical_accuracy(y_true, y_pred, padding_token)])

    train_inputs= pad_sequences(train_inputs, maxlen=(sequence_size), padding='pre', truncating='pre', value=padding_token)
    val_inputs= pad_sequences(val_inputs, maxlen=(sequence_size), padding='pre', truncating='pre', value=padding_token)
    train_targets= pad_sequences(train_targets, maxlen=(sequence_size), padding='pre', truncating='pre', value=padding_token)
    val_targets= pad_sequences(val_targets, maxlen=(sequence_size), padding='pre', truncating='pre', value=padding_token)

    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets)).batch(32).prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets)).batch(32).prefetch(tf.data.AUTOTUNE)

    callbacks = []
    if checkpoint_filepath:
        model_checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_filepath,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        callbacks.append(model_checkpoint_callback)
    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks, verbose=verbose)
    
    return model

# Load model weights from a checkpoint file
def load_model_weights(model, checkpoint_filepath):
    """
        Load model weights from a checkpoint file.
        model: Keras Model
        checkpoint_filepath: str, path to the checkpoint file
        Returns: model with loaded weights
    """
    if os.path.exists(checkpoint_filepath):
        model.load_weights(checkpoint_filepath)
        print(f"Model weights loaded from {checkpoint_filepath}")
    else:
        print(f"Checkpoint file {checkpoint_filepath} does not exist.")
    return model


def generate_recommendations(model, sequence, course_encoder, max_sequence_length, k, mask_token, padding_token=0):
    """
    Generates k course recommendations based on a user's course sequence.

    Args:
        model: The trained transformer model.
        sequence: A list of encoded course IDs representing the user's history.
        course_encoder: The LabelEncoder fitted on the course names.
        max_sequence_length: The maximum length of input sequences for the model.
        padding_token: The ID used for padding sequences.
        mask_token: The ID used for masking tokens in the sequence.
        k: The number of recommendations to generate.

    Returns:
        A list of k recommended course names.
        A list of k recommended course token.
    """
    current_sequence = list(sequence)
    recommended_courses_names = []
    recommended_courses_token = []

    for _ in range(k):

        current_sequence.append(mask_token)  # Append a single mask_token for iterative prediction

        # Prepare the current sequence for model input considering mask_token
        input_seq_padded = pad_sequences([current_sequence], maxlen=max_sequence_length, padding='pre', truncating='pre', value=padding_token)[0]
        # Predict the next course or k courses
        predictions = model.predict(np.array([input_seq_padded]), verbose=0)

        last_token_pred = predictions[0][-1]
        predicted_index = np.argmax(last_token_pred)

        # Decode the predicted course index
        # Ensure the predicted index is within the range of actual courses
        if predicted_index < course_encoder.classes_.shape[0]:
            predicted_course_encoded = predicted_index
            predicted_course_name = course_encoder.inverse_transform([predicted_course_encoded])[0]
        else:
            print(f"Warning: Predicted index {predicted_index} is out of bounds for course encoder vocabulary size {course_encoder.classes_.shape[0]}. Skipping recommendation.")

        # Add the predicted course (encoded ID) to the current sequence for the next iteration
        current_sequence.append(predicted_course_encoded)

        # Store the decoded recommended course name
        recommended_courses_names.append(predicted_course_name)
        recommended_courses_token.append(predicted_course_encoded)

    return recommended_courses_names, recommended_courses_token


def generate_recommendations_test_dataset(model, encoded_train_sequences, encoded_test_sequences, course_encoder, mask_token, max_sequence_length, k):
    """
    Evaluates the transformer model on the test set.

    Args:
        model: The trained transformer model.
        encoded_train_sequences: Encoded training sequences.
        encoded_test_sequences: Encoded testing sequences.
        course_encoder: The LabelEncoder fitted on the course names.
        max_sequence_length: The maximum length of input sequences for the model.
        mask_token: The ID used for masking tokens in the sequence.
        k: The number of recommendations to generate and evaluate.

    Returns:
        A list with the courses suggested or recommended.
        A list of the true courses viewed by the user.
    """

    # Iterate through users present in both train and test sets
    users = encoded_test_sequences.index

    courses_recommended_list = []
    courses_test_dataset = []
    num_evaluated_users = 0 # Initialize the counter


    print(f"Evaluating model on {len(users)} common users...")

    for user_id in tqdm(users):
        train_sequence = encoded_train_sequences.loc[user_id]
        test_sequence = encoded_test_sequences.loc[user_id]

        # We only evaluate if the test sequence has at least one item (the target)
        # and the training sequence has at least one item to base the recommendation on.
        if len(train_sequence) > 0 and len(test_sequence) > 0:
            num_evaluated_users += 1
            current_sequence = list(train_sequence)

            # Generate k recommendations based on the user's training sequence
            _, recommended_courses_encoded = generate_recommendations(model, current_sequence, course_encoder, max_sequence_length, k, mask_token=mask_token)
            courses_recommended_list.append(recommended_courses_encoded)
            courses_test_dataset.append(test_sequence) 

    print(f"Evaluated {num_evaluated_users} users.")

    return courses_recommended_list, courses_test_dataset
