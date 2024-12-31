import torch
import numpy as np

class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __getitem__(self, index):
        data = self.dataset[index]
        if isinstance(data, (tuple, list)):
            return *data, index
        elif isinstance(data, (dict,)):
            data["index"] = index
            return data
    
    def __len__(self):
        return len(self.dataset)
    
    def __repr__(self):
        return f"IndexDataset({repr(self.dataset)})"

class JoinDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_d):
        super().__init__()
        self.dataset_d = dataset_d

        self._key = list(dataset_d.keys())

        self._cnt = [ len(dataset_d[k]) for k in self._key ]
        self._sum = np.cumsum(self._cnt)
        self._len = self._sum[-1]
    
    def __getitem__(self, index):
        data = None
        for key, size, end in zip(self._key, self._cnt, self._sum):
            # beg <= index < end
            # beg <= index < beg + size
            # 0 <= index - beg < size
            # -size <= index - end < 0
            if index - end < 0:
                data = self.dataset_d[key][index+size-end] # index - beg = index - (end - size)
                break
        # assert data is not None, f"{self._key} {self._cnt} {self._sum} {index}" 

        if isinstance(data, (tuple, list)):
            return *data, index
        elif isinstance(data, (dict,)):
            data["index"] = index
            return data
        else:
            raise IndexError
    
    def __len__(self):
        return self._len

    def __repr__(self):
        return f"JoinDataset({repr(self.dataset_d)})"