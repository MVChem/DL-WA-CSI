import os

import torch
from torch.utils.data import Dataset
from mrspy.util import load_img

class MRSDataset(Dataset):
    def __init__(self, txt_file):
        """
        Args:
            txt_file (str): Path to the text file containing image folder paths.
        """
        self.txt_file = txt_file
        self.folder_paths = self._load_folder_paths()
        

    def _load_folder_paths(self):
        """Loads folder paths from the text file. Assumes paths in txt_file are full paths."""
        folder_paths = []
        with open(self.txt_file, 'r') as f:
            for line in f:
                folder_path = line.strip()
                folder_paths.append(folder_path)
        return folder_paths

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        """
        Reads data for a single folder path, performs batching and simulation.

        Args:
            idx (int): Index of the folder path to load.

        Returns:
            dict: water img and glu img (256 * 256)
        """
        folder_path = self.folder_paths[idx]

        water_img = load_img(os.path.join(folder_path, "water.jpg"), output_type="tensor", dtype=torch.float32)
        glu_img = load_img(os.path.join(folder_path, "glu.jpg"), output_type="tensor", dtype=torch.float32)
        lac_img = load_img(os.path.join(folder_path, "lac.jpg"), output_type="tensor", dtype=torch.float32)

        return water_img, glu_img, lac_img, folder_path
