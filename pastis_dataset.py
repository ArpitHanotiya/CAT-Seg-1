import os
import h5py
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

class PASTISDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        """
        Initialize PASTIS dataset.
        
        Args:
            data_dir (str): Path to the dataset directory containing S2_*.h5 files and ANNOTATIONS.pkl
            split (str): 'train', 'val', or 'test'
            transform: Optional transform to be applied to samples
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load metadata
        with open(os.path.join(data_dir, 'metadata.pkl'), 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Get patch IDs for the specified split
        self.patch_ids = self.metadata[split]
        
        # Load annotations
        with open(os.path.join(data_dir, 'ANNOTATIONS.pkl'), 'rb') as f:
            self.annotations = pickle.load(f)
    
    def __len__(self):
        return len(self.patch_ids)
    
    def __getitem__(self, idx):
        patch_id = self.patch_ids[idx]
        
        # Load Sentinel-2 time series data
        with h5py.File(os.path.join(self.data_dir, f'S2_{patch_id}.h5'), 'r') as f:
            s2_data = np.array(f['S2'])  # Shape: (time_steps, height, width, channels)
        
        # Load annotations
        semantic_mask = self.annotations[patch_id]['semantic']  # Shape: (height, width)
        
        # Convert to PyTorch tensors
        s2_data = torch.from_numpy(s2_data).float()
        s2_data = s2_data.permute(3, 0, 1, 2)  # Reshape to (channels, time_steps, height, width)
        s2_data = s2_data / 10000.0  # Normalize reflectance values
        semantic_mask = torch.from_numpy(semantic_mask).long()
        
        # Apply transforms if any
        if self.transform:
            s2_data = self.transform(s2_data)
        
        return {
            "file_name": f"S2_{patch_id}.h5",
            "image": s2_data,
            "sem_seg": semantic_mask,
            "height": s2_data.shape[2],
            "width": s2_data.shape[3]
        }
    
    def get_dicts(self):
        """
        Return dataset in Detectron2 format.
        """
        dataset_dicts = []
        for idx in range(len(self)):
            record = self[idx]
            dataset_dicts.append(record)
        return dataset_dicts