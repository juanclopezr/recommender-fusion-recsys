import pickle as pkl
import numpy as np
import os
import torch
import pandas as pd
import sys


from tqdm import tqdm

sys.path.append('/home/jcsanguino10/local_citation_model/Secuencial SR')
from evaluation_metrics import calculate_average_mrr, calculate_average_precision_at_k, calculate_average_ndcg_at_k, calculate_average_custom_precision_at_k

sys.path.append('/home/jcsanguino10/local_citation_model/recommender-fusion-recsys/loaders')
from create_dataloader_sequential import (load_course_encoder)

sys.path.append('/home/jcsanguino10/local_citation_model/recommender-fusion-recsys/architectures/Multimodal')
from multimodal import Autoencoder, MultimodalModel

if torch.cuda.is_available():
    # Change to just a particular GPU changing the enviroment variable
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    device = torch.device("cuda")
    # Change to just use a particular GPU via torch
    #torch.cuda.set_device("cuda:3")
    print(torch.cuda.current_device())

# Paths from the label encoder dicts 
PATH_TO_LABEL_ENCODER = '/home/jcsanguino10/local_citation_model/data/processed/'

# Path to folder with datasets
PATH_TO_DATASETS = '/home/jcsanguino10/local_citation_model/data/'

# Path to folder with checkpoints best models
PATH_TO_CHECKPOINTS = '/home/jcsanguino10/local_citation_model/models/'

label_encoder, dicts = load_course_encoder('/home', PATH_TO_LABEL_ENCODER)

df_binary = pd.read_pickle(f'{PATH_TO_DATASETS}train_binary_all_vectors_128_01_transe_seqvec.pkl')
df_bpr = pd.read_pickle(f'{PATH_TO_DATASETS}train_bpr_all_vectors_128_01_transe_seqvec.pkl')

df_test_binary = pd.read_pickle(f'{PATH_TO_DATASETS}test_binary_all_vectors_128_01_transe_seqvec.pkl')

df_test_recommender = df_test_binary[['user_id', 'full_item_seq']].drop_duplicates(subset='user_id').reset_index(drop=True)

# Generate a mapping of all user IDs to the list of course IDs they have taken using the full_item_seq column in df_bpr_df
user_courses_taken = {}
for _, row in df_binary.iterrows():
    user_id = row['user_id']
    courses = row['full_item_seq']
    user_courses_taken[user_id] = courses

df_test_recommender['courses_taken'] = df_test_recommender['user_id'].map(user_courses_taken)

def concat_columns_to_tensor(df, columns, new_column_name):
    """
    Concatenates specified columns in a DataFrame and creates a tensor.
    The resulting tensor is saved in a new column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): List of column names to concatenate.
        new_column_name (str): Name of the new column to store the tensor.

    Returns:
        pd.DataFrame: The DataFrame with the new column containing tensors.
    """
    df[new_column_name] = df[columns].apply(
        lambda row: torch.tensor([item for col in columns for item in row[col]], dtype=torch.float),
        axis=1
    )
    return df

def train_autoencoder_and_extract_encoder(data, input_dim, encoding_dims, epochs=50, lr=1e-3, 
                                         save_path=None, device='cuda', verbose=True):
    """
    Train an autoencoder.
    
    Args:
        data (torch.Tensor): Training data tensor of shape (batch_size, input_dim)
        input_dim (int): Dimension of input features
        encoding_dims (list): List of hidden layer dimensions for encoder
                             Example: [512, 256, 128] for 3-layer encoder
        epochs (int): Number of training epochs
        lr (float): Learning rate
        save_path (str): Path to save the best autoencoder model (optional)
        device (str): Device to train on ('cpu' or 'cuda')
        verbose (bool): Whether to print training progress
        
    Returns:
        encoder (nn.Module): The trained autoencoder model
    """
    
    if verbose:
        print(f"Starting autoencoder training...")
        print(f"Input dimension: {input_dim}")
        print(f"Encoding dimensions: {encoding_dims}")
        print(f"Final encoding dimension: {encoding_dims[-1]}")
        print(f"Training data shape: {data.shape}")
    
    # Create autoencoder
    autoencoder = Autoencoder(input_dim=input_dim, encoding_dims=encoding_dims)
    
    if verbose:
        print(f"Autoencoder architecture created")
        print(f"Encoder layers: {len(autoencoder.encoder)}")
        print(f"Decoder layers: {len(autoencoder.decoder)}")
    
    # Train the autoencoder using the enhanced train_autoencoder method
    trained_autoencoder = Autoencoder.train_autoencoder(
        autoencoder=autoencoder,
        data=data,
        epochs=epochs,
        lr=lr,
        save_path=save_path,
        device=device
    )
    
    
    if verbose:
        print(f" Training completed!")
        print(f" Encoder extracted successfully")
        print(f" Encoder output dimension: {encoding_dims[-1]}")
    
    return trained_autoencoder

def generate_recommendations_per_user(df, model, courses_dict, k=5, batch_size=64):

    user_tensors = [torch.tensor(x) if not isinstance(x, torch.Tensor) else x 
                for x in df["user_full_embeddings"].values]

    user_tensors = torch.stack(user_tensors)

    all_user_embs = user_tensors  # shape [num_users, dim]

    courses_already_taken = df["courses_taken"].values
    recommendations = []

    for i in tqdm(range(0, len(all_user_embs), batch_size), desc="Generating recommendations"):
        batch = all_user_embs[i:i+batch_size]
        batch_courses_taken = courses_already_taken[i:i+batch_size]
        # Generate recommendations for the batch
        batch_recs = model.generate_k_recommendations(courses_dict, batch, batch_courses_taken, k=k)
        recommendations.extend(batch_recs)

    df["recommendations"] = recommendations
    return df

def test_model(df, models, courses_dict, k):
    for model in models:
        temp_df = generate_recommendations_per_user(df.drop_duplicates(subset="user_id"), model, courses_dict)
        k=5
        courses_test_dataset = temp_df["full_item_seq"].to_list()
        courses_recommended_list = temp_df["recommendations"].to_list()

        avg_mrr = calculate_average_mrr(courses_test_dataset, courses_recommended_list)
        avg_ndcg_at_k = calculate_average_ndcg_at_k(courses_test_dataset, courses_recommended_list, k)
        avg_precision_at_k = calculate_average_precision_at_k(courses_test_dataset, courses_recommended_list, k)
        avg_custom_precision_at_k = calculate_average_custom_precision_at_k(courses_test_dataset, courses_recommended_list, k)
        return avg_mrr, avg_ndcg_at_k, avg_precision_at_k, avg_custom_precision_at_k

class BPRDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_feat = self.df.iloc[idx]['user_full_embeddings']
        pos_course_feat = self.df.iloc[idx]['pos_course_full_embeddings']
        neg_course_feat = self.df.iloc[idx]['neg_course_full_embeddings']
        return {
            'user': torch.tensor(user_feat, dtype=torch.float),
            'course_positive': torch.tensor(pos_course_feat, dtype=torch.float),
            'course_negative': torch.tensor(neg_course_feat, dtype=torch.float)
        }
    
class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_feat = self.df.iloc[idx]['user_full_embeddings']
        course_feat = self.df.iloc[idx]['course_full_embeddings']
        label = self.df.iloc[idx]['label']
        return {
            'user': torch.tensor(user_feat, dtype=torch.float),
            'course_positive': torch.tensor(course_feat, dtype=torch.float),
            'targets': torch.tensor(label, dtype=torch.float)
        }

def test_multimodal_model(columns_combinations, encoding_dims, shared_dimensions, layers_per_modality, regularization_bpr_values=[0.1], epochs=30, k=5, batch_size=64, pickle_results_path=None, verbose=True):

    #caculate the  total number of combinations to use tqdm
    total_combinations = len(columns_combinations) * 2 * (1 + len(regularization_bpr_values))
    print(f"Total combinations to test: {total_combinations}")

    #start process bar for tqdm
    pbar = tqdm(total=total_combinations)
    results = []
    for columns_to_concat_courses, columns_to_concat_users in columns_combinations:
        if verbose:
            print(f"Testing combination: Courses - {columns_to_concat_courses}, Users - {columns_to_concat_users}")
        else:
            pbar.set_description(f"Courses - {columns_to_concat_courses}, Users - {columns_to_concat_users}")
            sys.stdout = open(os.devnull, 'w')
        # Concatenate columns to create full embeddings
        df_binary_temp = concat_columns_to_tensor(df_binary, columns_to_concat_courses, 'course_full_embeddings')
        df_binary_temp = concat_columns_to_tensor(df_binary_temp, columns_to_concat_users, 'user_full_embeddings')

        # For BPR DataFrame the prefix pos and neg is added to the columns_to_concat lists
        df_bpr_temp = concat_columns_to_tensor(df_bpr, [f'pos_{col}' for col in columns_to_concat_courses], 'pos_course_full_embeddings')
        df_bpr_temp = concat_columns_to_tensor(df_bpr_temp, [f'neg_{col}' for col in columns_to_concat_courses], 'neg_course_full_embeddings')
        df_bpr_temp = concat_columns_to_tensor(df_bpr_temp, columns_to_concat_users, 'user_full_embeddings')

        # Create tensors for courses and users
        course_tensor = [torch.tensor(x, dtype=torch.float) for x in df_binary_temp['course_full_embeddings'].values]
        embeddings_course_tensor = torch.stack(course_tensor)

        user_tensor = [torch.tensor(x, dtype=torch.float) for x in df_binary_temp['user_full_embeddings'].values]
        embeddings_user_tensor = torch.stack(user_tensor)
        
        # Define a unique saving path based on the columns used
        saving_path = PATH_TO_CHECKPOINTS + '_'.join([col.split('_')[1] for col in columns_to_concat_users])

        course_encoder = train_autoencoder_and_extract_encoder(embeddings_course_tensor, embeddings_course_tensor.shape[1], encoding_dims, save_path=f'{saving_path}_encoder_course.pth' ,epochs=100, lr=1e-3, verbose=False)

        user_encoder = train_autoencoder_and_extract_encoder(embeddings_user_tensor, embeddings_user_tensor.shape[1], encoding_dims, save_path=f'{saving_path}_encoder_user.pth', epochs=100, lr=1e-3, verbose=False)

        modality_dims = {
            'course': embeddings_course_tensor.shape[1],
            'user': embeddings_user_tensor.shape[1]
        }

        # Create combinations parameters for the multimodal model (with/without autoencoder fusion) method and with BPR or Binary loss then for each combination create the model and test it.   

        combinations_user_bpr = [True, False]
        combinations_fusion_method = ['concat', 'by_autoencoder']

        for use_bpr in combinations_user_bpr:
            for fusion_method in combinations_fusion_method:
                temp_regularization_bpr_values = regularization_bpr_values if use_bpr else [0.0]
                for reg_lambda in temp_regularization_bpr_values:
                    print(f"Testing model with use_bpr={'bpr' if use_bpr else 'binary'} and fusion_method={fusion_method}")
                    if fusion_method == 'by_autoencoder':
                        print(f"Using autoencoder fusion method with encoding dimension {encoding_dims[-1]}")
                        model = MultimodalModel(modality_dims, use_bpr=use_bpr, fusion_method=fusion_method,shared_dim=shared_dimensions, layers_per_modality=layers_per_modality ,autoencoders={'course': course_encoder, 'user': user_encoder}, autoencoder_output_dim=encoding_dims[-1])
                    else:
                        print(f"Using concatenation fusion method")
                        #Concat the encoding dim and the layers per modality in a single list
                        new_layers_per_modality = encoding_dims + layers_per_modality
                        model = MultimodalModel(modality_dims, use_bpr=use_bpr, fusion_method=fusion_method,shared_dim=shared_dimensions, layers_per_modality=new_layers_per_modality ,autoencoders=None, autoencoder_output_dim=None)

                    if use_bpr:
                        dataset = BPRDataset(df_bpr_temp)
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                        model_path = f'{saving_path}_multimodal_{fusion_method}_{reg_lambda}_bpr.pth'
                    else:
                        dataset = BinaryDataset(df_binary_temp)
                        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                        model_path = f'{saving_path}_multimodal_{fusion_method}_binary.pth'
                    
                    model.train_model(
                        train_loader=dataloader,
                        epochs=epochs,
                        lr=1e-3,
                        device='cuda',
                        save_path=model_path,
                        reg_lambda=reg_lambda,
                        verbose=False
                    )

                    # Create a mapping of user_id to user_sequence_embedding from the train dataset
                    user_sequence_mapping = df_binary_temp.drop_duplicates(subset="user_id").set_index('user_id')['user_full_embeddings'].to_dict()
                    # Replace the user_sequence_embedding in the test dataset using the mapping
                    df_test_recommender['user_full_embeddings'] = df_test_recommender['user_id'].map(user_sequence_mapping)

                    course_sequence_mapping = df_bpr_temp.drop_duplicates(subset="pos_item_id").set_index('pos_item_id')['pos_course_full_embeddings'].to_dict()
                    
                    avg_mrr, avg_ndcg_at_k, avg_precision_at_k, avg_custom_precision_at_k = test_model(df_test_recommender, models=[model], courses_dict=course_sequence_mapping, k=k)

                    ## add the results
                    results.append({
                        'models_used': '_'.join([col.split('_')[1] for col in columns_to_concat_users]),
                        'use_bpr': 'bpr' if use_bpr else 'binary',
                        'fusion_method': fusion_method,
                        'reg_lambda': reg_lambda,
                        'avg_mrr': avg_mrr,
                        'avg_ndcg_at_k': avg_ndcg_at_k,
                        'avg_precision_at_k': avg_precision_at_k,
                        'avg_custom_precision_at_k': avg_custom_precision_at_k
                    })
                    #update progress bar
                    sys.stdout = sys.__stdout__
                    pbar.update(1)
    #Save results in a pkl file as a list of dicts
    if pickle_results_path is not None:
        with open(pickle_results_path, 'wb') as f:
            pkl.dump(results, f)
    else:
        with open(f'{PATH_TO_DATASETS}_multimodal_search_results.pkl', 'wb') as f:
            pkl.dump(results, f)
    return results