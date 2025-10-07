from torch.utils.data import Dataset, DataLoader
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from sklearn.preprocessing import LabelEncoder
import os


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


class SequentialDataset(Dataset):
    """
    Dataset class for Sequential training data
    """
    def __init__(self, sequences, max_sequence_length, mask_token, mask_prob=0.2, 
                 padding_token=0, mode="MASKED", sequential_model="BERT4Rec", min_sequence_length=4):
        self.sequences = sequences
        self.max_sequence_length = max_sequence_length
        self.mask_token = mask_token
        self.mask_prob = mask_prob
        self.padding_token = padding_token
        self.mode = mode
        self.sequential_model = sequential_model
        self.min_sequence_length = min_sequence_length
        
        # Process sequences and create input/target pairs
        if sequential_model == "BERT4Rec":
            self.inputs, self.targets = self._create_masked_sequences()
        else:
            self.inputs, self.targets = self._create_sequence_pairs()

    def _create_sequence_pairs(self):
        """Create input and target pairs for Sequential Transformer training"""
        expanded_sequences = []
        
        if self.mode == "AUGMENT":
            for seq in self.sequences:
                if len(seq) > self.min_sequence_length:
                    for length in range((len(seq) - self.min_sequence_length)):
                        sub_seq = seq[0:length + self.min_sequence_length]
                        expanded_sequences.append(sub_seq)
                else:
                    expanded_sequences.append(seq)
            # Shuffle the expanded sequences to ensure randomness
            random.shuffle(expanded_sequences)
            sequences = expanded_sequences
        else:
            sequences = self.sequences
        
        inputs, targets = [], []
        for sequence in sequences:
            if len(sequence) > 1:  # Need at least 2 items (input + target)
                inputs.append(sequence[:-1])  # All but last item as input
                targets.append([sequence[-1]])  # Last item as target
        
        return inputs, targets

    def _create_masked_sequences(self):
        """Create masked input and target sequences for BERT4Rec training"""
        expanded_sequences = []
        
        if self.mode == "MIXED":
            for seq in self.sequences:
                if len(seq) > self.min_sequence_length:
                    for length in range((len(seq) - self.min_sequence_length)):
                        sub_seq = seq[0:length + self.min_sequence_length]
                        expanded_sequences.append(sub_seq)
                else:
                    expanded_sequences.append(seq)
            # Shuffle the expanded sequences to ensure randomness
            random.shuffle(expanded_sequences)
            sequences = expanded_sequences
        else:
            sequences = self.sequences
        
        inputs, targets = [], []
        for seq in sequences:
            initial_size = min(len(seq), self.max_sequence_length)
            seq = seq[:initial_size]
            masked_input = list(seq)
            # index 0 is reserved for padding
            target = [0] * initial_size
            masked_count_tokens = 0
            
            for i in range(initial_size):
                if random.random() < self.mask_prob and masked_count_tokens < initial_size:
                    masked_count_tokens += 1
                    target[i] = masked_input[i]
                    masked_input[i] = self.mask_token
                if i == initial_size - 1 and masked_count_tokens == 0:
                    target[i] = masked_input[i]
                    masked_input[i] = self.mask_token
            
            inputs.append(masked_input)
            targets.append(target)
        
        return inputs, targets
    
    def _pad_sequence(self, seq):
        """Pad sequence to max_sequence_length"""
        if len(seq) >= self.max_sequence_length:
            return seq[:self.max_sequence_length]
        else:
            # Pad at the beginning (pre-padding)
            padding_length = self.max_sequence_length - len(seq)
            return [self.padding_token] * padding_length + seq
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_seq = self.inputs[idx]
        target_seq = self.targets[idx]
        
        # Pad sequences
        input_seq = self._pad_sequence(input_seq)
        if self.sequential_model == "BERT4Rec":
            target_seq = self._pad_sequence(target_seq)
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)


class CreateDataloaderSequential(object):
    """
    Construct Dataloaders for Sequential model
    """
    def __init__(self, args, user_sequences, course_encoder=None, dataset_path="./data", 
                 test_split=0.2, validation_split=0.1):
        """
        Args:
            args: Arguments object containing hyperparameters
            user_sequences: pd.Series where index is user_id and values are lists of course sequences
            course_encoder: LabelEncoder for courses (optional, will create if not provided)
            dataset_path: Path to save/load processed data
            test_split: Fraction of users to use for testing
            validation_split: Fraction of remaining users to use for validation
        """
        self.batch_size = args.batch_size
        self.max_sequence_length = getattr(args, 'max_sequence_length', None)
        self.mask_prob = getattr(args, 'mask_prob', 0.2)
        self.mode = getattr(args, 'sequence_mode', 'MASKED')  # 'MASKED' or 'MIXED'
        self.min_sequence_length = getattr(args, 'min_sequence_length', 4)
        self.dataset_path = dataset_path
        self.sequential_model = getattr(args, 'sequential_model', 'BERT4Rec')  # 'BERT4Rec' or other
        self.seed = getattr(args, 'seed', 42)
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        self.user_sequences = user_sequences
        self.course_encoder = course_encoder
        
        # Process and split data
        self._process_sequences()

        self._split_data(test_split, validation_split)
        
        # Create mask token (vocab_size)
        self.mask_token = len(self.course_encoder.classes_)
        self.padding_token = 0
    
    def _process_sequences(self):
        """Process and encode sequences"""
        if self.course_encoder is None:
            
            # Create course encoder
            all_courses = []
            for seq in self.user_sequences:
                all_courses.extend(seq)
            unique_courses = np.unique(all_courses)
            
            self.course_encoder = LabelEncoder()
            self.course_encoder.fit(unique_courses)
            
            # Add padding token
            self.course_encoder.classes_ = np.insert(self.course_encoder.classes_, 0, "<PAD>")
        
        # Encode sequences
        self.encoded_sequences = []
        for user_id, sequence in self.user_sequences.items():
            if len(sequence) >= 2:  # Need at least 2 items for training
                try:
                    encoded_seq = self.course_encoder.transform(sequence).tolist()
                    self.encoded_sequences.append((user_id, encoded_seq))
                except ValueError as e:
                    print(f"Error encoding sequence for user {user_id}: {e}")
                    continue
    
    def _split_data(self, test_split, validation_split):
        """Split data into train/validation/test"""
        # Shuffle sequences
        random.shuffle(self.encoded_sequences)
        
        n_total = len(self.encoded_sequences)
        n_test = int(n_total * test_split)
        n_val = int((n_total - n_test) * validation_split)

        test_data = self.encoded_sequences[-n_test:]
        val_data = self.encoded_sequences[-(n_test + n_val):-n_test]
        train_data = self.encoded_sequences[:-(n_test + n_val)]

        if self.max_sequence_length is None:
            # Estimate max_sequence_length from training data as the quantile 0.95
            max_size = int(np.percentile([len(seq) for _, seq in train_data], 95))
            self.max_sequence_length = max_size

        # Extract just the sequences (not user_ids for training)
        
        self.train_sequences = [seq for _, seq in train_data]
        self.val_sequences = [seq for _, seq in val_data]
        self.test_sequences = [seq for _, seq in test_data]
        
        # Keep user_ids for evaluation
        self.test_user_sequences = {user_id: seq for user_id, seq in test_data}
        self.val_user_sequences = {user_id: seq for user_id, seq in val_data}
    
    def get_train_dataloader(self):
        """Create training dataloader"""
        train_dataset = SequentialDataset(
            sequences=self.train_sequences,
            max_sequence_length=self.max_sequence_length,
            mask_token=self.mask_token,
            mask_prob=self.mask_prob,
            padding_token=self.padding_token,
            mode=self.mode,
            min_sequence_length=self.min_sequence_length,
            sequential_model=self.sequential_model
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
    
    def get_val_dataloader(self):
        """Create validation dataloader"""
        val_dataset = SequentialDataset(
            sequences=self.val_sequences,
            max_sequence_length=self.max_sequence_length,
            mask_token=self.mask_token,
            mask_prob=self.mask_prob,
            padding_token=self.padding_token,
            mode="MASKED",  # Use MASKED mode for validation
            min_sequence_length=self.min_sequence_length,
            sequential_model=self.sequential_model
        )
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
    def get_test_dataloader(self):
        """Create test dataloader for evaluation"""
        test_dataset = SequentialDataset(
            sequences=self.test_sequences,
            max_sequence_length=self.max_sequence_length,
            mask_token=self.mask_token,
            mask_prob=self.mask_prob,
            padding_token=self.padding_token,
            mode="MASKED",  # Use MASKED mode for testing
            min_sequence_length=self.min_sequence_length,
            sequential_model=self.sequential_model
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
    
   
    def save_processed_data(self, save_path=None):
        """Save processed data for future use"""

        if save_path is None and self.sequential_model == "BERT4Rec":
            save_path = os.path.join(self.dataset_path, 'bert4rec_processed_data.pkl')
        elif save_path is None and self.sequential_model != "BERT4Rec":
            save_path = os.path.join(self.dataset_path, 'sequential_processed_data.pkl')
        
        data_to_save = {
            'train_sequences': self.train_sequences,
            'val_sequences': self.val_sequences,
            'test_sequences': self.test_sequences,
            'test_user_sequences': self.test_user_sequences,
            'val_user_sequences': self.val_user_sequences,
            'course_encoder': self.course_encoder,
            'mask_token': self.mask_token,
            'padding_token': self.padding_token,
            'max_sequence_length': self.max_sequence_length
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Processed data saved to {save_path}")
    
    @classmethod
    def load_processed_data(cls, args, load_path):
        """Load previously processed data"""
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a dummy instance
        instance = cls.__new__(cls)
        instance.batch_size = args.batch_size
        instance.dataset_path = os.path.dirname(load_path)
        
        # Load saved data
        instance.train_sequences = data['train_sequences']
        instance.val_sequences = data['val_sequences']
        instance.test_sequences = data['test_sequences']
        instance.test_user_sequences = data['test_user_sequences']
        instance.val_user_sequences = data['val_user_sequences']
        instance.course_encoder = data['course_encoder']
        instance.mask_token = data['mask_token']
        instance.padding_token = data['padding_token']
        instance.max_sequence_length = data['max_sequence_length']
        
        # Set other attributes from args
        instance.mask_prob = getattr(args, 'mask_prob', 0.2)
        instance.mode = getattr(args, 'sequence_mode', 'MASKED')
        instance.min_sequence_length = getattr(args, 'min_sequence_length', 4)
        
        print(f"Processed data loaded from {load_path}")
        return instance
    
    def get_statistics(self):
        """Get dataset statistics"""
        all_sequences = self.train_sequences + self.val_sequences + self.test_sequences
        sequence_lengths = [len(seq) for seq in all_sequences]
        
        stats = {
            'num_users': len(self.encoded_sequences),
            'num_items': len(self.course_encoder.classes_) - 1,  # Exclude padding token
            'total_sequences': len(all_sequences),
            'avg_sequence_length': np.mean(sequence_lengths),
            'min_sequence_length': np.min(sequence_lengths),
            'max_sequence_length': np.max(sequence_lengths),
            'median_sequence_length': np.median(sequence_lengths),
            'vocab_size': len(self.course_encoder.classes_)
        }
        
        return stats


# Utility functions for integration with existing codebase
def create_sequential_dataloader(args, df=None, max_sequence_size=None, user_sequences=None, course_encoder=None):
    """
    Function to create sequential dataloader
    
    Args:
        args: Arguments object
        df: DataFrame with columns ['user_id', 'course_name', 'timestamp'] (optional)
        user_sequences: Pre-processed user sequences (optional)
        course_encoder: Pre-trained course encoder (optional)
    
    Returns:
        CreateDataloaderSequential instance
    """
    if user_sequences is None:
        if df is None:
            raise ValueError("Either df or user_sequences must be provided")
        
        # Preprocess data to create sequences
        
        user_sequences = preprocess_data(df)
    
    return CreateDataloaderSequential(
        args=args,
        user_sequences=user_sequences,
        course_encoder=course_encoder,
        dataset_path=getattr(args, 'dataset_path', './data')
    )
