import os
from PIL import Image
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]
        self.transform = transform


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.data_dir, image_name)
        
        # Load the image using PIL
        image = Image.open(image_path).convert("RGB")  # Keep the image in color (RGB)

        # Classify by name
        if "NFF" in image_name:
            label = 0
        elif "AFF" in image_name:
            label = 1
        else:
            label = -1

        # Resize the image to 256x256 pixels
        if self.transform:
            image = self.transform(image)

        return image, label

print('Done!')
