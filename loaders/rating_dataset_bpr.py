from torch.utils.data import Dataset
from torch import tensor
import torch

class RatingDataset(Dataset):
    def __init__(self, user_list, item_preferred_list, item_not_preferred_list, tokenization_list=None):
        super(RatingDataset, self).__init__()
        self.user_list = user_list
        self.item_preferred_list = item_preferred_list
        self.item_not_preferred_list = item_not_preferred_list
        self.tokenization_list = tokenization_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item_preferred = self.item_preferred_list[idx]
        item_not_preferred = self.item_not_preferred_list[idx]
        if self.tokenization_list is not None:
            return (
                tensor(user, dtype=torch.long),
                tensor(item_preferred, dtype=torch.long),
                tensor(rating_not_preferred, dtype=torch.long),
                self.tokenization_list[idx]
                )

        return (
            tensor(user, dtype=torch.long),
            tensor(item_preferred, dtype=torch.long),
            tensor(item_not_preferred, dtype=torch.long)
            )
            