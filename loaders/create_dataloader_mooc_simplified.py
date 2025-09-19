from loaders.rating_dataset import RatingDataset
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
    def __init__(self, args, train_ratings, test_ratings, dataset_path, graph_embeddings=None, num_users=6863):
        self.num_ng = args.num_ng
        self.num_ng_test = args.num_ng_test
        self.batch_size = args.batch_size
        self.dataset_path = dataset_path

        self.NUM_USERS = num_users#694529

        self.graph_embeddings = graph_embeddings

        if not os.path.exists(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl'):
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
        if self.graph_embeddings is not None:
            graph_embeddings = torch.zeros((len(batch), self.graph_embeddings.shape[1] * 2))
            for i, x in enumerate(batch):
                user = x[0].item()
                item = x[1].item() + self.NUM_USERS
                graph_embeddings[i] = torch.tensor(np.concatenate([self.graph_embeddings[user], self.graph_embeddings[item]]))

        return (
            torch.stack([x[0] for x in batch]),
            torch.stack([x[1] for x in batch]),
            torch.stack([x[2] for x in batch]),
            graph_embeddings if self.graph_embeddings is not None else None
        )

    def get_train_instance(self):
        if not os.path.exists (f'{self.dataset_path}/train_users_{self.num_ng}.pkl'):
            users, items, ratings = [], [], []
            train_ratings = pd.merge(self.train_ratings, self.negatives[['user_id', 'train_negative_samples']], on='user_id')
            for row in tqdm(train_ratings.itertuples(), total=train_ratings.shape[0]):
                users.append(int(row.user_id))
                items.append(int(row.item_id))
                ratings.append(float(row.rating))
                for i in getattr(row, 'train_negative_samples'):
                    users.append(int(row.user_id))
                    items.append(int(i))
                    ratings.append(float(0))
            pickle.dump(users, open(f'{self.dataset_path}/train_users_{self.num_ng}.pkl', 'wb'))
            pickle.dump(items, open(f'{self.dataset_path}/train_items_{self.num_ng}.pkl', 'wb'))
            pickle.dump(ratings, open(f'{self.dataset_path}/train_ratings_{self.num_ng}.pkl', 'wb'))
        else:
            #print('Checking!!!!')
            users = pickle.load(open(f'{self.dataset_path}/train_users_{self.num_ng}.pkl', 'rb'))
            items = pickle.load(open(f'{self.dataset_path}/train_items_{self.num_ng}.pkl', 'rb'))
            ratings = pickle.load(open(f'{self.dataset_path}/train_ratings_{self.num_ng}.pkl', 'rb'))

        dataset = RatingDataset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        #print('max of items', max(items))
            
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0, collate_fn=self.collate_fn)


    def get_test_instance(self):
        if not os.path.exists(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl'):
            users, items, ratings = [], [], []
            test_ratings = pd.merge(self.test_ratings, self.negatives[['user_id', 'test_negative_samples']], on='user_id')
            for row in tqdm(test_ratings.itertuples(), total= test_ratings.shape[0]):
                users.append(int(row.user_id))
                items.append(int(row.item_id))
                ratings.append(float(row.rating))
                for i in getattr(row, 'test_negative_samples'):
                    users.append(int(row.user_id))
                    items.append(int(i))
                    ratings.append(float(0))
            pickle.dump(users, open(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl', 'wb'))
            pickle.dump(items, open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'wb'))
            pickle.dump(ratings, open(f'{self.dataset_path}/test_ratings_{self.num_ng_test}.pkl', 'wb'))
        else:
            users = pickle.load(open(f'{self.dataset_path}/test_users_{self.num_ng_test}.pkl', 'rb'))
            items = pickle.load(open(f'{self.dataset_path}/test_items_{self.num_ng_test}.pkl', 'rb'))
            ratings = pickle.load(open(f'{self.dataset_path}/test_ratings_{self.num_ng_test}.pkl', 'rb'))

        dataset = RatingDataset(
            user_list=users,
            item_list=items,
            rating_list=ratings)
        
        return DataLoader(dataset, batch_size=self.num_ng_test+1, shuffle=False, num_workers=0, collate_fn=self.collate_fn)