from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pandas as pd


class TrafficDataset2(Dataset):
    def __init__(self, root):
        self.dataset = pd.read_csv(root)  # raw data
        self.features = self.dataset.iloc[:, :-2].values  # (3569, 240)
        self.features = self._preProcess(self.features)
        self.labels = self.dataset.iloc[:, -2].values  # (3569,)
        self.classes = np.unique(self.labels)
        # self.dates = data[:, -1]

    def _preProcess(self, data, wd=3):
        # data: sample_size x 720, ndarray

        # Apply moving average for noise removal along each row
        # for i in range(data.shape[0]):
        #     for j in range(data.shape[1]):
        #         left, right = max(0, j - wd), min(data.shape[1], j + wd)
        #         data[i, j] = np.round(np.mean(data[i, left:right]))

        # Normalize the data
        # data = (data - np.mean(data)) / np.std(data)
        data = (data - np.min(data)) / (np.max(data) - np.min(data))

        return data
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # idx is a single integer index.
        x, y = self.features[idx], self.labels[idx]
        
        # in DataLoader, pls use 'for batch_idx, (item_q, item_k, label) in enumerate(train_loader):' 
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class TrafficDataset(Dataset):
    def __init__(self, root):
        self.dataset = pd.read_csv(root)  # raw data
        self.features = self.dataset.iloc[:, :-2].values  # (1000, 720)
        self.features = self._preProcess(self.features)
        self.labels = self.dataset.iloc[:, -2].values  # (1000,)
        self.classes = np.unique(self.labels)
        # self.dates = data[:, -1]

    def _preProcess(self, data, wd=3):
        # data: sample_size x 720, ndarray

        # fill missing values with the adjacent values for each sample/row
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if np.isnan(data[i, j]):
                    left, right = j - 1, j + 1
                    while left >= 0 and np.isnan(data[i, left]):
                        left -= 1
                    
                    while right < data.shape[1] and np.isnan(data[i, right]):
                        right += 1
                    
                    if left >= 0 and right < data.shape[1]:
                        fill_value = (data[i, left] + data[i, right]) / 2
                    elif left < 0 and right >= data.shape[1]:
                        raise ValueError("All values are missing")
                    elif left < 0:
                        fill_value = data[i, right]
                    elif right >= data.shape[1]:
                        fill_value = data[i, left]
                    
                    data[i, j] = fill_value

        # Apply moving average for noise removal along each row
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                left, right = max(0, j - wd), min(data.shape[1], j + wd)
                data[i, j] = np.round(np.mean(data[i, left:right]))

        # Normalize the data
        data = (data - np.mean(data)) / np.std(data)

        return data
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # idx is a single integer index.
        item_q, label = self.features[idx], self.labels[idx]

        # Get the indices of the samples that belong to the same group as the current sample
        group_indices = np.where(self.labels == label)[0]
        
        # Randomly select an index from the group indices
        k_idx = np.random.choice(group_indices, size=2).tolist()
        k_idx = k_idx[0] if k_idx[0] != idx else k_idx[1]

        # Get the corresponding feature for batch_k
        item_k = self.features[k_idx]
        
        # in DataLoader, pls use 'for batch_idx, (item_q, item_k, label) in enumerate(train_loader):' 
        return torch.tensor(item_q, dtype=torch.float32), torch.tensor(item_k, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
