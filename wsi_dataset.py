import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from torch.utils.data.distributed import DistributedSampler
import pandas as pd


class WSIDataset(Dataset):
    def __init__(self, df, feat_dir, label_dict):
        super(WSIDataset, self).__init__()

        self.df = df
        self.labels = df["label"]
        self.feat_dir = feat_dir
        self.label_dict = label_dict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, id):
        file_name = self.df['image_id'].values[id]
        pt_dir = os.path.join(self.feat_dir, f"{file_name}.pt")

        feat = torch.load(pt_dir).float()
        if self.label_dict is not None:
            label = self.label_dict[self.labels[id]]
        else:
            label = self.labels[id]
        label = torch.tensor(label).float()

        return feat, label.long()


class WSIDataModule(LightningDataModule):
    def __init__(self, config, split_k=0, dist=True):
        super(WSIDataModule, self).__init__()
        """
        prepare datasets and samplers
        """
        df = pd.read_csv(config["Data"]["dataframe"])

        train_index = df[df["fold"] != split_k].index
        train_df = df.loc[train_index].reset_index(drop=True)
        val_index = df[df["fold"] == split_k].index
        val_df = df.loc[val_index].reset_index(drop=True)
        test_df = pd.read_csv(config["Data"]["test_df"])
        dfs = [train_df, val_df, test_df]    # get training, test and validation datasets

        self.dist = dist
        self.label_dict = config["Data"]["label_dict"]

        self.datasets = [WSIDataset(df,
                                    config["Data"]["feat_dir"],
                                    config["Data"]["label_dict"]) for df in dfs]

        self.batch_size = config["Data"]["batch_size"]
        self.num_workers = config["Data"]["num_workers"]

    def setup(self, stage):
        if self.dist:
            self.samplers = [DistributedSampler(dataset, shuffle=True)
                             for dataset in self.datasets]
        else:
            self.samplers = [None, None, None]

    def train_dataloader(self):
        loader = DataLoader(
            self.datasets[0],
            batch_size=self.batch_size,
            sampler=self.samplers[0],
            num_workers=self.num_workers,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.datasets[1],
            batch_size=self.batch_size,
            sampler=self.samplers[1],
            num_workers=self.num_workers,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.datasets[2],
            batch_size=self.batch_size,
            sampler=self.samplers[2],
            num_workers=self.num_workers,
        )
        return loader
