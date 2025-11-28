import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils import rgb_to_lab, hard_encode_ab

class ColorizationDataset(Dataset):
    """
    Dataset para colorización en modo CLASIFICACIÓN (importante).
    - Input: canal L (luminosidad) normalizado.
    - Target: índice de bin de color por píxel (hard).
    """

    def __init__(self, image_paths, pts_ab, img_size=256, transform=None):
        """
        Args:
            image_paths: lista de paths a imágenes RGB.
            pts_ab: tensor (Q, 2) con los centros de los bins (en Lab).
            img_size: tamaño al que se reescala la imagen.
        """
        self.image_paths = image_paths
        self.pts_ab = pts_ab
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Cargar imagen RGB
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")

        # Aplicación de transform (en caso de ser especificado)
        if self.transform is not None:
            img = self.transform(img)
        else:
            # Sino, reescalado a tamaño fijo
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)

        # Pasar a numpy [0,1]
        rgb_np = np.array(img).astype(np.float32) / 255.0

        # Convertir a Lab
        lab_np = rgb_to_lab(rgb_np)
        L_np = lab_np[:, :, 0]
        ab_np = lab_np[:, :, 1:]

        # Convertir a tensores
        L_tensor = torch.from_numpy(L_np).unsqueeze(0)
        L_tensor = L_tensor.float() / 100.0     # L en [0,1]
        ab_tensor = torch.from_numpy(ab_np).permute(2, 0, 1).float()

        q_idx = hard_encode_ab(ab_tensor, self.pts_ab)

        return L_tensor, q_idx.long()