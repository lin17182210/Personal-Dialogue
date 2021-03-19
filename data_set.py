from torch.utils.data import Dataset


class DialogueDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        input_ids = self.data_list[idx]
        return input_ids
