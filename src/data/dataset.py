import os
import pickle
import polars as pl
import numpy as np
from torch.utils.data import Dataset
import torch
from PIL import Image

class BCCDDataset(Dataset):
    def __init__(self, df: pl.DataFrame, image_base: str, transforms = None):
        super().__init__()
        self.df = df
        self.image_base = image_base
        self.transforms = transforms

        self.image_ids = df["file_id"].unique().to_list()
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        _sub_df = self.df.filter(pl.col("file_id") == image_id)
        image_data = np.array(Image.open(os.path.join(self.image_base, _sub_df["image_file"][0])).convert("RGB"))
        
        boxes, labels = self._get_bbbox_labels(_sub_df)
        
        if self.transforms is not None:
            transformed = self.transforms(
                image=image_data,
                bboxes=boxes,
                labels=labels
            )
            image_data  = transformed['image']    # C x H x W tensor
            boxes  = transformed['bboxes']   # list(x1, y1, x2, y2)
            labels = transformed['labels']
        
        if len(boxes) == 0:
            boxes_tensor  = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,),   dtype=torch.int64)
            area_tensor   = torch.zeros((0,),   dtype=torch.float32)
        else:
            boxes_tensor  = torch.tensor(boxes,  dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            area_tensor   = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * \
                            (boxes_tensor[:, 3] - boxes_tensor[:, 1])
        
        target = {
            'boxes':    boxes_tensor,
            'labels':   labels_tensor,
            'area':     area_tensor,
            'image_id': torch.tensor([idx]),
            # iscrowd = 1 means the annotation covers a crowd of objects
            # and should be ignored during mAP computation.
            # BCCD has no crowd annotations, so this is always 0.
            'iscrowd':  torch.zeros((len(labels_tensor),), dtype=torch.int64)
        }

        return image_data, target
    
    def _get_bbbox_labels(self, df: pl.DataFrame):
        exp1 = (pl.col("xmin") < pl.col("xmax")).alias("xvalid")
        exp2 = (pl.col("ymin") < pl.col("ymax")).alias("yvalid")
        exp3 = (pl.col("xvalid") & pl.col("yvalid")).alias("valid")
        exp4 = pl.col("valid") == True
        
        boxes = []
        labels = []
        
        _sub_df = df.with_columns([
            exp1,
            exp2
        ]).with_columns(exp3).filter(exp4)
        
        for row in _sub_df.rows(named=True):
            boxes.append([row["xmin"], row["ymin"], row["xmax"], row["ymax"]])
            labels.append(row["label"])
        return boxes, labels


def collate_fn(batch):
    images  = [item[0] for item in batch] # this will be list of tensors
    targets = [item[1] for item in batch] # this will ne list of dicts: key -> tensor
    return images, targets