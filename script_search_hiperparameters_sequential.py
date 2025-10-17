# Import Required Libraries
import os
import sys
import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
import torch
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set up paths
sys.path.append('/home/jcsanguino10/local_citation_model/recommender-fusion-recsys')

# Import Sequential Transformer modules
import architectures.Sequence.sec_transformer_pytorch as seq_transformer
import architectures.Sequence.bert4rec_pytorch as bert4rec

# Import Sequential Transformer dataloader modules 
from loaders.create_dataloader_sequential import (
    CreateDataloaderSequential, preprocess_data, create_sequential_dataloader, load_course_encoder
)

from evaluation_metrics import calculate_average_mrr, calculate_average_precision_at_k, calculate_average_ndcg_at_k, calculate_average_custom_precision_at_k


print("✅ All libraries imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    # Change to just a particular GPU changing the enviroment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")
    # Change to just use a particular GPU via torch
    #torch.cuda.set_device("cuda:3")
    print(torch.cuda.current_device())


# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

train_dataset = '/home/jcsanguino10/local_citation_model/datasets/GCF/dataTrain.csv'
test_dataset = '/home/jcsanguino10/local_citation_model/datasets/GCF/dataTest.csv'

course_encoder_path = '/home/jcsanguino10/local_citation_model/data/id2entity.pkl'

course_encoder_path_saved = '/home/jcsanguino10/local_citation_model/data/processed'

# Load and concatenate datasets for preprocessing
train_df = pd.read_csv(train_dataset)
test_df = pd.read_csv(test_dataset)
# df = pd.concat([train_df, test_df])
# df.reset_index(drop=True, inplace=True)

user_sequences_test = preprocess_data(test_df)

# Sequence length distribution (after preprocessing)
user_sequences = preprocess_data(train_df)
sequence_lengths = user_sequences.apply(len)

# Model Configuration and Hyperparameters
class Args:
    """Arguments class for model configuration"""

    def __init__(self):
        # Data parameters
        self.batch_size = 32
        self.max_sequence_length = None  # Set None to estimate from data
        self.sequence_mode = 'AUGMENT'  # 'AUGMENT' or 'NO_AUGMENT' for Sequential Transformer
        self.min_sequence_length = 4
        self.sequential_model = 'SEQUENTIAL'  # 'SEQUENTIAL' or 'BERT4REC'
        self.test_split = 0.2
        self.validation_split = 0.1
        self.dataset_path = '/home/jcsanguino10/local_citation_model/data'
        self.padding_token = 0  # Padding token index
        
        # Model parameters
        self.embedding_dim = 128
        self.num_heads = 8
        self.ff_dim = 512
        self.num_transformer_blocks = 4
        self.dropout_rate = 0.1
        
        # Training parameters
        self.epochs = 30
        self.learning_rate = 1e-4
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 42
        
        # Evaluation parameters
        self.k_recommendations = 5
        
        # Model saving
        self.model_save_path = '/home/jcsanguino10/local_citation_model/models'
        self.checkpoint_path = None  # Will be set dynamically

# Grid search parameters
num_heads_options = [1, 2, 4]  # Number of attention heads in the transformer
num_transformer_blocks_options = [1, 3, 5]  # Number of transformer blocks
model_types = ['SEQUENTIAL', 'BERT4Rec']  # Models to evaluate
sequential_modes = ['AUGMENT', 'NO_AUGMENT']  # Sequential Transformer modes
bert4rec_modes = ['MASKED', 'MIXED']  # BERT4Rec modes

# Create hyperparameter combinations
hyperparameter_combinations = []

for num_heads in num_heads_options:
    for num_blocks in num_transformer_blocks_options:
        for model_type in model_types:
            if model_type == 'SEQUENTIAL':
                for mode in sequential_modes:
                    hyperparameter_combinations.append({
                        'num_heads': num_heads,
                        'num_transformer_blocks': num_blocks,
                        'model_type': model_type,
                        'mode': mode
                    })
            elif model_type == 'BERT4Rec':
                for mode in bert4rec_modes:
                    hyperparameter_combinations.append({
                        'num_heads': num_heads,
                        'num_transformer_blocks': num_blocks,
                        'model_type': model_type,
                        'mode': mode
                    })

print(f"Total hyperparameter combinations to evaluate: {len(hyperparameter_combinations)}")

def run_experiment(user_sequences, experiment_config):
    """Run a single experiment with given hyperparameters"""
    
    # Create args object for this experiment
    args = Args()
    args.num_heads = experiment_config['num_heads']
    args.num_transformer_blocks = experiment_config['num_transformer_blocks']
    args.sequential_model = experiment_config['model_type']
    args.sequence_mode = experiment_config['mode']

    
    # Set checkpoint path
    model_name = f"{args.sequential_model}_{args.sequence_mode}_{args.num_heads}h_{args.num_transformer_blocks}b"
    args.checkpoint_path = os.path.join(args.model_save_path, f"{model_name}_best.pth")
    
    # Ensure model directory exists
    os.makedirs(args.model_save_path, exist_ok=True)
    
    print(f"Running experiment: {model_name}")
    
    # Get the appropriate model utilities
    if args.sequential_model == 'BERT4Rec':
        model_utils = bert4rec
        args.mask_prob = 0.2
    else:
        model_utils = seq_transformer

    if os.path.exists(course_encoder_path):
        course_encoder, _ = load_course_encoder(course_encoder_path, course_encoder_path_saved)
        # Get padding token index from the LabelEncoder
        if "<PAD>" in course_encoder.classes_:
            args.padding_token = np.where(course_encoder.classes_ == "<PAD>")[0][0]
        else:
            #Add <PAD> token to the encoder without losing previous mappings
            args.padding_token = len(course_encoder.classes_)
            course_encoder.classes_ = np.append(course_encoder.classes_, "<PAD>")
            
        print(f"✅ Loaded course encoder from {course_encoder_path}. Vocab size: {len(course_encoder.classes_)}. Padding token index: {args.padding_token}")
    else:
        course_encoder = None
        #raise FileNotFoundError(f"Course encoder file not found at {course_encoder_path}")
        

    # Create the dataloader using our dedicated system
    dataloader_creator = create_sequential_dataloader(
        args=args,
        user_sequences=user_sequences, 
        course_encoder=course_encoder
    )

    # Get the dataloaders
    train_loader = dataloader_creator.get_train_dataloader()
    val_loader = dataloader_creator.get_val_dataloader()

    # Get course encoder for later use
    #check if the path exists

    vocab_size = len(course_encoder.classes_)

    # Create the model
    model = model_utils.create_model(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        padding_idx=args.padding_token,
        num_transformer_blocks=args.num_transformer_blocks,
        dropout_rate=args.dropout_rate
    )

    # Move model to device before training
    model = model.to(args.device)

    # Train the model using the dedicated train_model function
    trained_model = model_utils.train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_filepath=args.checkpoint_path,
        device=args.device,
        verbose=0  # Enable verbose output to see training progress
    )

    # Load best model weights from checkpoint
    model_utils.load_pytorch_weights(trained_model, args.checkpoint_path, device=args.device)
    print("✅ Loaded best model weights from checkpoint.")
    

    # Create train and test sequences for evaluation
    test_recommender_sequences = {}
    train_recommender_sequences = {}

    for user_id, seq in dataloader_creator.encoded_sequences:
        train_recommender_sequences[user_id] = seq

    # User course encoder to parse the test sequences
    for user_id, seq in user_sequences_test.items():
        # Encode using the course encoder, ignoring unknown courses
        encoded_seq = [course_encoder.transform([course])[0] for course in seq if course in course_encoder.classes_]
        test_recommender_sequences[user_id] = encoded_seq

    # Create pandas series from the dictionaries
    recommender_user_sequences = pd.Series(train_recommender_sequences)
    test_recommender_sequences = pd.Series(test_recommender_sequences)

    # print heads of sequences
    print(f"Sample training sequences:\n{recommender_user_sequences.shape}")
    print(f"Sample test sequences:\n{test_recommender_sequences.shape}")


    # Generate recommendations based on model type
    if args.sequential_model == 'BERT4Rec':
        # BERT4Rec needs mask_token
        mask_token = vocab_size  # Mask token is usually vocab_size (after all valid tokens)
        courses_recommended_list, courses_test_dataset = model_utils.generate_recommendations_test_dataset(
            trained_model, 
            recommender_user_sequences, 
            test_recommender_sequences, 
            course_encoder,
            mask_token,
            dataloader_creator.max_sequence_length, 
            args.k_recommendations, 
            args.device, 
            padding_token=args.padding_token
        )
    else:
        # Sequential Transformer
        courses_recommended_list, courses_test_dataset = model_utils.generate_recommendations_test_dataset(
            trained_model, 
            recommender_user_sequences, 
            test_recommender_sequences, 
            course_encoder,
            dataloader_creator.max_sequence_length, 
            args.k_recommendations, 
            args.device, 
            padding_token=args.padding_token
        )

    # Calculate evaluation metrics
    k = 5
    # Ensure there are items to evaluate
    if courses_test_dataset and courses_recommended_list and len(courses_test_dataset) == len(courses_recommended_list) and len(courses_test_dataset) > 0:
        # Calculate average metrics using the defined functions
        avg_mrr = calculate_average_mrr(courses_test_dataset, courses_recommended_list)
        avg_ndcg_at_k = calculate_average_ndcg_at_k(courses_test_dataset, courses_recommended_list, k)
        avg_precision_at_k = calculate_average_precision_at_k(courses_test_dataset, courses_recommended_list, k)
        avg_custom_precision_at_k = calculate_average_custom_precision_at_k(courses_test_dataset, courses_recommended_list, k)
        
        print(f"Experiment results: MRR={avg_mrr:.4f}, NDCG@{k}={avg_ndcg_at_k:.4f}, Precision@{k}={avg_precision_at_k:.4f}")
        
        return avg_mrr, avg_precision_at_k, avg_ndcg_at_k, avg_custom_precision_at_k, len(train_loader), len(val_loader)
    else:
        print("Warning: No valid evaluation data")
        return 0.0, 0.0, 0.0, 0.0, len(train_loader), len(val_loader)

def run_all_experiments(hyperparameter_combinations, user_sequences, results_filepath="grid_search_results_pytorch.csv", verbose=False):
    """Run all experiments for all combinations of hyperparameters and save the results in a CSV file"""
    results = []

    for i, experiment_config in enumerate(tqdm(hyperparameter_combinations, desc="Running experiments")):      
        print(f"\n=== Experiment {i+1}/{len(hyperparameter_combinations)} ===")  
        if not verbose: 
            sys.stdout = open(os.devnull, 'w')
        try:
            # Run the experiment
            mrr, precision_at_5, ndcg_at_5, custom_precision_at_5, train_samples, val_samples = run_experiment(
                user_sequences, 
                experiment_config
            )

            # Store the results
            results.append({
                "num_heads": experiment_config['num_heads'],
                "num_transformer_blocks": experiment_config['num_transformer_blocks'],
                "model": experiment_config['model_type'],
                "mode": experiment_config['mode'],
                "MRR": mrr,
                "Precision@5": precision_at_5,
                "NDCG@5": ndcg_at_5,
                "Custom Precision@5": custom_precision_at_5,
                "train_samples": train_samples,
                "val_samples": val_samples
            })
            
            # Save intermediate results to CSV after each experiment
            pd.DataFrame(results).to_csv(results_filepath, index=False)
            print(f"Saved intermediate results to {results_filepath}")
            
        except Exception as e:
            print(f"Error in experiment {i+1}: {str(e)}")
            # Save error result
            results.append({
                "num_heads": experiment_config['num_heads'],
                "num_transformer_blocks": experiment_config['num_transformer_blocks'],
                "model": experiment_config['model_type'],
                "mode": experiment_config['mode'],
                "MRR": 0.0,
                "Precision@5": 0.0,
                "NDCG@5": 0.0,
                "Custom Precision@5": 0.0,
                "train_samples": 0,
                "val_samples": 0,
                "error": str(e)
            })
            pd.DataFrame(results).to_csv(results_filepath, index=False)
            continue
        if not verbose:
            sys.stdout = sys.__stdout__

    # Save final results to CSV
    pd.DataFrame(results).to_csv(results_filepath, index=False)
    print(f"All experiments completed. Results saved to {results_filepath}")
    
    return results

if __name__ == "__main__":
    print("Starting PyTorch Grid Search for Sequential Recommendation Models")
    print("================================================================")
    
    # Run experiments for all hyperparameter combinations
    path_dataset = '/home/jcsanguino10/local_citation_model/data/sequential_recommender_results.csv'

    results = run_all_experiments(hyperparameter_combinations, user_sequences, results_filepath=path_dataset, verbose=False)
    
    # Display summary of results
    print("\n=== Grid Search Summary ===")
    results_df = pd.DataFrame(results)
    if 'error' not in results_df.columns:
        results_df['error'] = ''
    
    # Show best results by MRR
    valid_results = results_df[results_df['error'] == ''].copy()
    if len(valid_results) > 0:
        best_result = valid_results.loc[valid_results['MRR'].idxmax()]
        print(f"Best configuration:")
        print(f"Model: {best_result['model']}, Mode: {best_result['mode']}")
        print(f"Heads: {best_result['num_heads']}, Blocks: {best_result['num_transformer_blocks']}")
        print(f"MRR: {best_result['MRR']:.4f}, NDCG@5: {best_result['NDCG@5']:.4f}, Precision@5: {best_result['Precision@5']:.4f}")
    else:
        print("No successful experiments completed.")
    
    print(f"\nTotal experiments: {len(results)}")
    if 'error' in results_df.columns:
        failed_experiments = len(results_df[results_df['error'] != ''])
        print(f"Failed experiments: {failed_experiments}")
        print(f"Successful experiments: {len(results) - failed_experiments}")
    
    print("Grid search completed!")




