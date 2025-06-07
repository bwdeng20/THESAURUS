from torch.utils.data import Dataset

class SingleElementDataset(Dataset):
    def __init__(self, data):
        self.data = data  # Store your data

    def __len__(self):
        return 1  # Dataset contains a single element

    def __getitem__(self, idx):
        return self.data  # R