import os
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
from torch import nn
import numpy as np
from thop import profile
from PIL import Image
import shutil
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model.network_scunet import SCUNet
from model.network_MBRS import Network








class ImageProcessingDataset(Dataset):
    def __init__(self, root_dir):
        self.root = root_dir
        self.image_paths = []
        self.rel_dirs = []

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        for root, _, files in os.walk(root_dir):
            rel_dir = os.path.relpath(root, root_dir)
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(root, f))
                    self.rel_dirs.append(rel_dir)

    def load_model(self, path_encoder_decoder: str, path_discriminator: str):
        self.load_model_ed(path_encoder_decoder)
        self.load_model_dis(path_discriminator)

    def __len__(self):
        return len(self.image_paths)
    
    def generate_binary_seed(self, seed_str: str) -> int:
        seed_str = seed_str.lower().replace("\\", "/").split('/')[-1]
        seed_str = seed_str.split('.')[0]
        hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
        return int.from_bytes(hash_bytes[:4], byteorder="big")

    def generate_binary_data(self, seed: int, length: int = 30) -> list:
        rng = np.random.RandomState(seed)
        return rng.randint(0, 2, size=length).tolist()

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            seed = self.generate_binary_seed(img_path)
            binary_data = self.generate_binary_data(seed, 30)
            return self.transform(img), idx, torch.tensor(binary_data, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {str(e)}")
            return None, idx

def batch_process(model, dataloader, output_root, device):

    denormalize = transforms.Compose([
        transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2]),
    ])

    with torch.no_grad(), tqdm(total=len(dataloader)) as pbar:
        for batch in dataloader:
            inputs, indices, message = batch
            
            valid_inputs = inputs.to(device)
            valid_indices = indices
            
            if valid_inputs.shape[0] == 0:
                continue

            normalized, message = valid_inputs, message.to(device)
            
            outputs = model(normalized, message)
            
            denorm_outputs = denormalize(outputs.cpu().clamp(-1, 1))
            
            for tensor, idx in zip(denorm_outputs, valid_indices):
                orig_path = dataset.image_paths[idx]
                rel_dir = dataset.rel_dirs[idx]
                filename = os.path.basename(orig_path)
                
                output_dir = os.path.join(output_root, rel_dir)
                os.makedirs(output_dir, exist_ok=True)
                
                img = transforms.ToPILImage()(tensor)
                img.save(os.path.join(output_dir, filename))
            
            pbar.update(1)

if __name__ == "__main__":
    input_root = "/data/experiment/data/gtos128_all/train"
    output_root = "/data/experiment/model/HiDDeN/Hidden_35_gtos/train"
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    network = Network()
    EC_path = "D:\python\隐写鲁棒\MBRS\right\D_4_psnr_30.91484602877968.pth"
    D_path = "D:\python\隐写鲁棒\MBRS\right\EC_4.pth"
    network.load_model(EC_path, D_path)


    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    batch_process(network, dataloader, output_root, device)