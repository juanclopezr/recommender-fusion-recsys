from loaders.rating_dataset_bpr import RatingDataset
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
    def __init__(self, args, train_ratings, test_ratings, dataset_path, num_users=6863):
        self.num_ng = args.num_ng
        self.num_ng_test = args.num_ng_test
        self.batch_size = args.batch_size
        self.dataset_path = dataset_path

        self.NUM_USERS = num_users#694529

        if not os.path.exists(f'{self.dataset_path}/test_users_bpr_{self.num_ng_test}.pkl'):
            self.train_ratings = train_ratings
            self.test_ratings = test_ratings
            self.ratings = pd.concat([train_ratings, test_ratings], ignore_index=True)
            print(train_ratings.shape, test_ratings.shape, self.ratings.shape)
            self.user_pool = set(self.ratings['user_id'].unique())
            self.item_pool = set(self.ratings['item_id'].unique())
            print('negative sampling')
            self.negatives = self._negative_sampling(self.ratings)
            print('done')
            
        random.seed(args.seed)

    def _negative_sampling(self, ratings):
        interact_status = (
            ratings.groupby('user_id')['item_id']
            .apply(set)
            .reset_index()
            .rename(columns={'item_id': 'interacted_items'}))
        interact_status['train_negative_samples'] = interact_status['interacted_items'].apply(lambda x: random.sample(self.item_pool - x, self.num_ng))
        interact_status['test_negative_samples'] = interact_status['interacted_items'].apply(lambda x: random.sample(self.item_pool - x, self.num_ng_test))
        return interact_status[['user_id', 'train_negative_samples', 'test_negative_samples']]
    
    
    def collate_fn(self, batch):
        return (
            torch.stack([x[0] for x in batch]),
            torch.stack([x[1] for x in batch]),
            torch.stack([x[2] for x in batch]),
            #graph_embeddings if self.graph_embeddings is not None else None
        )

    def get_train_instance(self):
        if not os.path.exists (f'{self.dataset_path}/train_users_bpr_{self.num_ng}.pkl'):
            users, items_preferred, items_not_preferred = [], [], []
            train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'train_negative_samples']], on='user_id')
            for row in tqdm(train_ratings.itertuples(), total=train_ratings.shape[0]):
                users.append(int(row.user_id))
                items_preferred.append(int(row.item_id))
                items_not_preferred.append(int(random.sample(row.train_negative_samples, 1)[0]))
            pickle.dump(users, open(f'{self.dataset_path}/train_users_bpr_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_preferred, open(f'{self.dataset_path}/train_items_preferred_bpr_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_not_preferred, open(f'{self.dataset_path}/train_items_not_preferred_bpr_{self.num_ng}.pkl', 'wb'))
        else:
            #print('Checking!!!!')
            users = pickle.load(open(f'{self.dataset_path}/train_users_bpr_{self.num_ng}.pkl', 'rb'))
            items_preferred = pickle.load(open(f'{self.dataset_path}/train_items_preferred_bpr_{self.num_ng}.pkl', 'rb'))
            items_not_preferred = pickle.load(open(f'{self.dataset_path}/train_items_not_preferred_bpr_{self.num_ng}.pkl', 'rb'))

        dataset = RatingDataset(
            user_list=users,
            item_preferred_list=items_preferred,
            item_not_preferred_list=items_not_preferred)
        #print('max of items', max(items))
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)


    def get_test_instance(self):
        if not os.path.exists (f'{self.dataset_path}/test_users_bpr_{self.num_ng}.pkl'):
            users, items_preferred, items_not_preferred = [], [], []
            train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'test_negative_samples']], on='user_id')
            for row in tqdm(train_ratings.itertuples(), total=train_ratings.shape[0]):
                users.append(int(row.user_id))
                items_preferred.append(int(row.item_id))
                items_not_preferred.append(int(random.sample(row.train_negative_samples, 1)[0]))
            pickle.dump(users, open(f'{self.dataset_path}/test_users_bpr_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_preferred, open(f'{self.dataset_path}/test_items_preferred_bpr_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items_not_preferred, open(f'{self.dataset_path}/test_items_not_preferred_bpr_{self.num_ng}.pkl', 'wb'))
        else:
            #print('Checking!!!!')
            users = pickle.load(open(f'{self.dataset_path}/test_users_bpr_{self.num_ng}.pkl', 'rb'))
            items_preferred = pickle.load(open(f'{self.dataset_path}/test_items_preferred_bpr_{self.num_ng}.pkl', 'rb'))
            items_not_preferred = pickle.load(open(f'{self.dataset_path}/test_items_not_preferred_bpr_{self.num_ng}.pkl', 'rb'))

        dataset = RatingDataset(
            user_list=users,
            item_preferred_list=items_preferred,
            item_not_preferred_list=items_not_preferred)
        
        return DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=0, collate_fn=self.collate_fn)