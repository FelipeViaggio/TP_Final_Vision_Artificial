"""
Funciones auxiliares para el proyecto de colorización de imágenes.
"""

import torch
import numpy as np
from skimage import color
import matplotlib.pyplot as plt


# ============================================
# CONVERSIÓN DE ESPACIOS DE COLOR
# ============================================

def rgb_to_lab(img):
    """
    Convierte imagen RGB a espacio LAB.
    
    Args:
        img: numpy array (H, W, 3) en rango [0, 1]
    
    Returns:
        lab: numpy array (H, W, 3) con canales L, a, b
    """
    return color.rgb2lab(img)


def lab_to_rgb(lab):
    """
    Convierte imagen LAB a espacio RGB.
    
    Args:
        lab: numpy array (H, W, 3) con canales L, a, b
    
    Returns:
        rgb: numpy array (H, W, 3) en rango [0, 1]
    """
    return color.lab2rgb(lab)


def normalize_lab(lab):
    """
    Normaliza los canales LAB para entrenamiento.
    
    L: [0, 100] → [0, 1]
    a: [-128, 127] → [-1, 1]
    b: [-128, 127] → [-1, 1]
    
    Args:
        lab: numpy array (H, W, 3)
    
    Returns:
        lab_norm: numpy array (H, W, 3) normalizado
    """
    lab_norm = lab.copy()
    lab_norm[:, :, 0] = lab[:, :, 0] / 100.0  # L
    lab_norm[:, :, 1] = lab[:, :, 1] / 128.0  # a
    lab_norm[:, :, 2] = lab[:, :, 2] / 128.0  # b
    return lab_norm


def denormalize_lab(lab_norm):
    """
    Desnormaliza los canales LAB después de predicción.
    
    Args:
        lab_norm: numpy array (H, W, 3) normalizado
    
    Returns:
        lab: numpy array (H, W, 3) en rango original
    """
    lab = lab_norm.copy()
    lab[:, :, 0] = lab_norm[:, :, 0] * 100.0  # L
    lab[:, :, 1] = lab_norm[:, :, 1] * 128.0  # a
    lab[:, :, 2] = lab_norm[:, :, 2] * 128.0  # b
    return lab