# dataset.py
import pydicom
import numpy as np
from torch.utils.data import Dataset

class PE_Dataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        dicom_file = pydicom.dcmread(file_path)
        image = dicom_file.pixel_array.astype(np.float32)
        image = image * dicom_file.RescaleSlope + dicom_file.RescaleIntercept
        image = (image - image.min()) / (image.max() - image.min())
        if self.transform:
            image = self.transform(image)
        return image, label
