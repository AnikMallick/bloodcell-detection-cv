import os
import pickle
import polars as pl
import numpy as np
from torch.utils.data import Dataset
import torch

def read_pkl(path: str) -> dict:
    with open(path, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d 

def make_df(path: str, metadata: dict) -> pl.DataFrame:
    d = read_pkl(path)
    img_data = [np.dstack([img[i: i + 1024].reshape((32, 32)) for i in range(0, len(img), 1024)]).astype(np.uint8) for img in d[b'data']]
    _data_dict = {
        "label": d[b'labels'],
        "label_names": [metadata[b'label_names'][l].decode("utf-8") for l in d[b'labels']]
    }
    _df = pl.DataFrame(_data_dict)
    _df = _df.with_columns(pl.Series("data", img_data, dtype=pl.Object))
    return _df


def make_df2(path: str, metadata: dict) -> pl.DataFrame:
    d = read_pkl(path)
    # _data_dict = {"data": val for i, val in enumerate(d[b'data'])}
    _data_dict ={}
    _data_dict["label"] = d[b'labels']
    _data_dict["data"] = d[b'data']
    _data_dict["label_names"] = [metadata[b'label_names'][l].decode("utf-8") for l in d[b'labels']]
    _df = pl.DataFrame(_data_dict)
    return _df

def read_data(path: str) -> tuple[pl.DataFrame]:
    meta = os.path.join(path, "batches.meta")
    train_pref = "data_batch_"
    test_pref = "test"
    
    files = os.listdir(path)
    
    metadata = read_pkl(meta)
    
    train_data = []
    test_data = []
    for file in files:
        if file.startswith(train_pref):
            train_data.append(make_df(os.path.join(path, file), metadata))
        elif file.startswith(test_pref):
            test_data.append(make_df(os.path.join(path, file), metadata))
    
    return pl.concat(train_data, how="vertical"), pl.concat(test_data, how="vertical")

def read_data_v2(path: str) -> tuple[pl.DataFrame]:
    meta = os.path.join(path, "batches.meta")
    train_pref = "data_batch_"
    test_pref = "test"
    
    files = os.listdir(path)
    
    metadata = read_pkl(meta)
    
    train_data = []
    test_data = []
    for file in files:
        if file.startswith(train_pref):
            train_data.append(make_df2(os.path.join(path, file), metadata))
        elif file.startswith(test_pref):
            test_data.append(make_df2(os.path.join(path, file), metadata))
    
    return pl.concat(train_data, how="vertical"), pl.concat(test_data, how="vertical")


class CIFARData(Dataset):
    def __init__(self, df: pl.DataFrame):
        super().__init__()
        self.df = df
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        return {
            "data": torch.tensor(np.asarray(self.df[index]["data"].to_list()), dtype=torch.float32),
            "label": torch.tensor(np.asarray(self.df[index]["label"].to_list()), dtype=torch.long),
        }

def shift_data(val: list[int]):
    shift = 4
    size = 32
    
    new_data = []
    for i in range(0, len(val), size):
        new_data.extend((([0] * shift) + val[i : i + size])[:size])
    return new_data