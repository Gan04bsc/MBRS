import os
import json
import hashlib
import numpy as np
from pathlib import Path
import torch
from torch import nn
import numpy as np
from options import *
from thop import profile
from PIL import Image
import shutil
from model.hidden import Hidden
from noise_layers.noiser import Noiser
import kornia
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model.network_scunet import SCUNet


def psnr_ssim_acc(image, H_img, L_img):
    # psnr
    H_psnr = kornia.metrics.psnr(
        ((image + 1) / 2).clamp(0, 1),
        ((H_img.detach() + 1) / 2).clamp(0, 1),
        1,
    )
    L_psnr = kornia.metrics.psnr(
    ((image + 1) / 2).clamp(0, 1),
    ((L_img.detach() + 1) / 2).clamp(0, 1),
    1,
    )
    # ssim
    H_ssim = kornia.metrics.ssim(
        ((image + 1) / 2).clamp(0, 1),
        ((H_img.detach() + 1) / 2).clamp(0, 1),
        window_size=11,
    ).mean()
    L_ssim = kornia.metrics.ssim(
        ((image + 1) / 2).clamp(0, 1),
        ((L_img.detach() + 1) / 2).clamp(0, 1),
        window_size=11,
    ).mean()
    return H_psnr, L_psnr, H_ssim, L_ssim


def model_from_checkpoint(hidden_net, checkpoint):
    """ Restores the hidden_net object from a checkpoint object """
    hidden_net.encoder_decoder.load_state_dict(checkpoint['enc-dec-model'])
    hidden_net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'])
    hidden_net.discriminator.load_state_dict(checkpoint['discrim-model'])
    hidden_net.optimizer_discrim.load_state_dict(checkpoint['discrim-optim'])


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



if __name__ == "__main__":

    input_root = "/data/experiment/data/gtos128_all/val"
    batch_size = 32
    num_workers = 4

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    noise_config = []
    hidden_config = HiDDenConfiguration(H=128, W=128,
                                        message_length=30,
                                        encoder_blocks=4, encoder_channels=64,
                                        decoder_blocks=7, decoder_channels=64,
                                        use_discriminator=True,
                                        use_vgg=False,
                                        discriminator_blocks=3, discriminator_channels=64,
                                        decoder_loss=1,
                                        encoder_loss=0.7,
                                        adversarial_loss=1e-3,
                                        enable_fp16=False
                                        )
    noiser = Noiser(noise_config, device)
    model = Hidden(hidden_config, device, noiser, None)

    checkpoint = torch.load('/data/experiment/model/HiDDeN/runs/30_gtos_I_psnr 2025.03.22--12-31-28/checkpoints/30_gtos_I_psnr--epoch-180--psnr-35.13345308805767.pyt')
    model_from_checkpoint(model, checkpoint)
    encoder = model.encoder_decoder.encoder
    decoder = model.encoder_decoder.decoder
    encoder.eval()
    decoder.eval()

    scunet = SCUNet(in_nc=3,config=[4,4,4,4,4,4,4],dim=64)

    # scunet.load_state_dict(torch.load('/data/experiment/model/SCUNet/runs/gtos_GN_75-2025-04-01-18:31-train/checkpoint/gtos_GN_75--epoch-25.pth')['network'], strict=True)
    scunet.load_state_dict(torch.load('/data/experiment/model/SCUNet/model_zoo/scunet_color_real_psnr.pth'), strict=True)
    scunet.to(device)
    scunet.eval()

    dataset = ImageProcessingDataset(input_root)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    sigma = [0, 15, 25, 50, 75]
    clean = 0
    for n in sigma:
        bitwise_avg_err_n_history = []
        bitwise_avg_err_r_history = []
        oH_psnrs = []
        oL_psnrs = []
        iH_psnrs = []
        iL_psnrs = []
        N_psnrs = []
        with torch.no_grad():
            for data in dataloader:
                inputs, indices, message = data

                noise = torch.Tensor(np.random.normal(0, n, inputs.shape)/128.).to(device)
                
                inputs = inputs.to(device)

                message = message.to(device)

                output_img = encoder(inputs, message)
                output_img_n = output_img + noise
                output_img_r = scunet((output_img_n+1)/2)

                decoded_messages_n = decoder(output_img_n)
                decoded_messages_r = decoder(output_img_r*2 - 1)
                
                decoded_rounded_n = decoded_messages_n.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err_n = np.sum(np.abs(decoded_rounded_n - message.detach().cpu().numpy())) / (
                        batch_size * 30)

                decoded_rounded_r = decoded_messages_r.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err_r = np.sum(np.abs(decoded_rounded_r - message.detach().cpu().numpy())) / (
                        batch_size * 30)
                oH_psnr, oL_psnr, _ , _ = psnr_ssim_acc(output_img.cpu(), output_img_r.cpu(), output_img_n.cpu())
                iH_psnr, iL_psnr, _ , _ = psnr_ssim_acc(inputs.cpu(), output_img_r.cpu(), output_img_n.cpu())
                N_psnr, _, _ , _ = psnr_ssim_acc(inputs.cpu(), (noise + inputs).cpu(), output_img_n.cpu())
                oH_psnrs.append(oH_psnr)
                oL_psnrs.append(oL_psnr)
                iH_psnrs.append(iH_psnr)
                iL_psnrs.append(iL_psnr)
                N_psnrs.append(N_psnr)
                bitwise_avg_err_n_history.append(bitwise_avg_err_n)
                bitwise_avg_err_r_history.append(bitwise_avg_err_r)
        noise = 1 - np.mean(bitwise_avg_err_n_history)
        recover = 1 - np.mean(bitwise_avg_err_r_history)
        if n == 0:
            clean = 1 - np.mean(bitwise_avg_err_n_history)
        else:
            revover_rate = (recover - noise) / (clean - noise)
            print('in sigma {}, recovery rate {:.4f}'.format(n, revover_rate * 100))
        imporve_rate = (recover - noise) / noise
        print('in sigma {}, increase rate {:.4f}'.format(n, imporve_rate * 100))
        print('in sigma {}, nosie image accuracy {:.4f}'.format(n, noise * 100))
        print('in sigma {}, recover image accuracy {:.4f}'.format(n, recover * 100))
        print('in sigma {}, H_psnr_wm_to_r {:.4f}'.format(n, np.mean(oH_psnrs)))
        print('in sigma {}, L_psnr_wm_to_n {:.4f}'.format(n, np.mean(oL_psnrs)))
        print('in sigma {}, H_psnr_ori_to_r {:.4f}'.format(n, np.mean(iH_psnrs)))
        print('in sigma {}, L_psnr_ori_to_n {:.4f}'.format(n, np.mean(iL_psnrs)))
        print('in sigma {}, N_psnr {:.4f}'.format(n, np.mean(N_psnrs)))