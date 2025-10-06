from loaders.rating_dataset_mixed_independent import RatingDataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding
import torch
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
import os

class CreateDataloader(object):
    """
    Construct Dataloaders
    """
    def __init__(self, args, train_ratings, test_ratings, dataset_path, graph_embeddings, bpr_embeddings, num_users=6863):
        self.num_ng = args.num_ng
        self.num_ng_test = args.num_ng_test
        self.batch_size = args.batch_size
        self.dataset_path = dataset_path

        self.NUM_USERS = num_users
        
        self.graph_embeddings = graph_embeddings
        self.bpr_user_embeddings = bpr_user_embeddings
        self.bpr_item_embeddings = bpr_item_embeddings

        if not os.path.exists(f'{self.dataset_path}/test_users_mixed_independent_{self.num_ng_test}.pkl'):
            self.train_ratings = train_ratings
            self.test_ratings = test_ratings
            self.ratings = pd.concat([train_ratings, test_ratings], ignore_index=True)
            print(train_ratings.shape, test_ratings.shape, self.ratings.shape)
            self.user_pool = set(self.ratings['user_id'].unique())
            self.item_pool = set(self.ratings['item_id'].unique())
            print('negative sampling')
            self.negatives = self._negative_sampling()
            print('done')
        self.user_text_vectors = self.calculate_user_text_vectors()
            
        random.seed(args.seed)

    def _negative_sampling(self):
        interact_status = (
            self.ratings.groupby('user_id')['item_id']
            .apply(set)
            .reset_index()
            .rename(columns={'item_id': 'interacted_items'}))
        interact_status['train_negative_samples'] = interact_status['interacted_items'].apply(lambda x: random.sample(self.item_pool - x, self.num_ng))
        interact_status['test_negative_samples'] = interact_status['interacted_items'].apply(lambda x: random.sample(self.item_pool - x, self.num_ng_test))
        return interact_status[['user_id', 'train_negative_samples', 'test_negative_samples']]
    
    def calculate_user_text_vectors(self):
        text_vectors = self.train_ratings.groupby('user_id')['text_embedding'].mean().to_dict()
        return text_vectors
    
    def calculate_item_text_vectors(self):
        text_vectors = self.train_ratings.groupby('item_id')['text_embedding'].first().to_dict()
        return text_vectors
    
    def get_graph_embedding(self, entity_id, entity_type='user'):
        if entity_type == 'user':
            return self.graph_embeddings[entity_id, :]
        return self.graph_embeddings[entity_id + self.NUM_USERS, :]
    
    def get_bpr_embedding(self, entity_id, entity_type='user'):
        if entity_type == 'user':
            return self.bpr_user_embeddings[entity_id, :]
        return self.bpr_item_embeddings[entity_id, :]
    
    def collate_fn(self, batch):
        return (
            torch.stack([x[0] for x in batch]),
            torch.stack([x[1] for x in batch]),
            torch.stack([x[2] for x in batch]),
            #graph_embeddings if self.graph_embeddings is not None else None
        )

    def get_train_instance(self):
        if not os.path.exists (f'{self.dataset_path}/train_users_mixed_independent_{self.num_ng}.pkl'):
            users, items_preferred, items_not_preferred = [], [], []
            train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'train_negative_samples']], on='user_id')
            for row in tqdm(train_ratings.itertuples(), total=train_ratings.shape[0]):
                users.append(np.concat([get_graph_embedding(int(row.user_id)), get_bpr_embedding(int(row.user_id)), self.user_text_vectors[int(row.user_id)]]))
                items_preferred.append(np.concat([get_graph_embedding(int(row.item_id), entity_type='item'), get_bpr_embedding(int(row.item_id), entity_type='item'), row.text_embedding]))
                items_not_preferred.append(np.concat([get_graph_embedding(int(random.sample(row.train_negative_samples, 1)[0]), entity_type='item'), get_bpr_embedding(int(random.sample(row.train_negative_samples, 1)[0]), entity_type='item'), self.item_text_vectors[int(random.sample(row.train_negative_samples, 1)[0])]]))
            pickle.dump(users, open(f'{self.dataset_path}/train_users_mixed_independent_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_preferred, open(f'{self.dataset_path}/train_items_preferred_mixed_independent_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_not_preferred, open(f'{self.dataset_path}/train_items_not_preferred_mixed_independent_{self.num_ng}.pkl', 'wb'))
        else:
            #print('Checking!!!!')
            users = pickle.load(open(f'{self.dataset_path}/train_users_mixed_independent_{self.num_ng}.pkl', 'rb'))
            items_preferred = pickle.load(open(f'{self.dataset_path}/train_items_preferred_mixed_independent_{self.num_ng}.pkl', 'rb'))
            items_not_preferred = pickle.load(open(f'{self.dataset_path}/train_items_not_preferred_mixed_independent_{self.num_ng}.pkl', 'rb'))

        dataset = RatingDataset(
            user_list=users,
            item_preferred_list=items_preferred,
            item_not_preferred_list=items_not_preferred)
        #print('max of items', max(items))
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)


    def get_test_instance(self):
        if not os.path.exists (f'{self.dataset_path}/test_users_mixed_independent_{self.num_ng}.pkl'):
            users, items_preferred, items_not_preferred = [], [], []
            test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'test_negative_samples']], on='user_id')
            for row in tqdm(test_ratings.itertuples(), total=test_ratings.shape[0]):
                users.append(np.concat([get_graph_embedding(int(row.user_id)), get_bpr_embedding(int(row.user_id)), self.user_text_vectors[int(row.user_id)]]))
                items_preferred.append(np.concat([get_graph_embedding(int(row.item_id), entity_type='item'), get_bpr_embedding(int(row.item_id), entity_type='item'), row.text_embedding]))
                items_not_preferred.append(np.concat([get_graph_embedding(int(random.sample(row.test_negative_samples, 1)[0]), entity_type='item'), get_bpr_embedding(int(random.sample(row.test_negative_samples, 1)[0]), entity_type='item'), self.item_text_vectors[int(random.sample(row.test_negative_samples, 1)[0])]]))
            pickle.dump(users, open(f'{self.dataset_path}/test_users_mixed_independent_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_preferred, open(f'{self.dataset_path}/test_items_preferred_mixed_independent_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_not_preferred, open(f'{self.dataset_path}/test_items_not_preferred_mixed_independent_{self.num_ng}.pkl', 'wb'))
        else:
            #print('Checking!!!!')
            users = pickle.load(open(f'{self.dataset_path}/test_users_mixed_independent_{self.num_ng}.pkl', 'rb'))
            items_preferred = pickle.load(open(f'{self.dataset_path}/test_items_preferred_mixed_independent_{self.num_ng}.pkl', 'rb'))
            items_not_preferred = pickle.load(open(f'{self.dataset_path}/test_items_not_preferred_mixed_independent_{self.num_ng}.pkl', 'rb'))

        dataset = RatingDataset(
            user_list=users,
            item_preferred_list=items_preferred,
            item_not_preferred_list=items_not_preferred)
        #print('max of items', max(items))
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)