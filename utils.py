"""
Funciones auxiliares para el proyecto de colorización de imágenes.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from skimage import color


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


# ============================================
# PREPARACIÓN / ENTRENAMIENTO
# ============================================

def generate_ab_bins(grid_size=10, L_value=50):
    """
    Genera la grilla de puntos (a,b) con paso 'grid_size',
    filtra los puntos fuera del rango RGB y devuelve los puntos válidos.
    
    Returns:
        pts_ab: np.ndarray de shape (#valid_points, 2)
    """
    # Se define rango del espacio ab
    ab_range = np.arange(-110, 111, grid_size)
    pts_in = []

    # Recorrer la grilla e incluir en pts_in los puntos dentro del rango válido
    for a in ab_range:
        for b in ab_range:
            lab = np.array([L_value, a, b]).reshape(1, 1, 3)

            # Convertir a RGB para verificar que esté dentro del rango
            rgb = lab_to_rgb(lab)
            
            if (np.all(rgb >= 0.0) and np.all(rgb <= 1.0)):
                pts_in.append([a, b])
    
    pts_in = np.array(pts_in)
    
    return pts_in


def compute_color_class_weights(train_loader, num_classes, device, lambda_smooth=0.5, max_batches=None):
    """
    Calcula:
        - histogramas de bins de color en todo el train_loader
        - frecuencias p_q
        - pesos balanceados w_q
    
    Args:
        train_loader: DataLoader de entrenamiento.
        num_classes: número de bins Q.
        devivce: 'cuda' o 'cpu'.
        lambda_smooth: lambda para el suavizado de frecuencias.
        
    Returns:
        p_q: tensor (Q,) con frecuencias normalizadas.
        w_q: tensor (Q,) con pesos re-balanceados.
    """
    # Histograma
    hist = torch.zeros(num_classes, dtype=torch.float64)

    with torch.no_grad():
        for b_idx, (L_batch, q_batch) in enumerate(train_loader):
            if max_batches is not None and b_idx >= max_batches:
                break
            
            q_batch = q_batch.to(device)
            
            # Aplanar
            q_flat = q_batch.view(-1)

            # Contar ocurrencias de cada clase presente en el batch
            vals, counts = torch.unique(q_flat, return_counts=True)

            # Sumar al histograma
            hist[vals.cpu()] += counts.cpu().to(hist.dtype)
    
    # Frecuencias normalizadas
    p_q = hist / hist.sum()

    eps = 1e-8
    p_q = torch.clamp(p_q, min=eps)

    Q = float(num_classes)

    # Suavizado de frecuencias
    p_tilde = (1.0 - lambda_smooth) * p_q + (lambda_smooth / Q)

    # Pesos inversos
    w_q = 1.0 / p_tilde

    # Normalizar para que el peso medio sea 1
    norm = (w_q * p_q).sum()
    w_q = w_q / norm

    return p_q.float(), w_q.float()


def hard_encode_ab(ab_tensor, pts_ab):
    """
    Hard encoding de ab en bins.
    
    Args:
        ab_tensor: tensor de shape (2, H, W) con canales a,b en unidades Lab reales
        pts_ab: tensor de shape (Q, 2) con los centros de los bins
    
    Devuelve:
        q_idx: tensor de shape (H, W) con índices [0, Q-1]
    """
    # Verificar que el tensor venga en forma (H, W, 2)
    if ab_tensor.shape[0] == 2:
        ab_hw2 = ab_tensor.permute(1, 2, 0) # (H, W, 2)
    else:
        ab_hw2 = ab_tensor

    H, W, _ = ab_hw2.shape
    ab_flat = ab_hw2.reshape(-1, 2) # (N, 2), N = H*W

    # Asegurar que esté todo en float
    ab_flat = ab_flat.float()
    pts_ab = pts_ab.to(ab_flat.device).float()  # (Q, 2). Adapta el tensor dependiendo si se usa CPU/GPU

    # Calcular las distancias euclídeas entre píxel-bin (N, Q) y guardar la menor distnacia
    dists = torch.cdist(ab_flat.unsqueeze(0), pts_ab.unsqueeze(0)).squeeze(0)
    q_flat = torch.argmin(dists, dim=1)

    q_idx = q_flat.view(H, W).long()
    
    return q_idx


def build_soft_encoding_matrix(pts_ab, T=0.5, K=5):
    """
    Construye una matriz (Q, Q) donde cada fila i es una distribución
    suave sobre las Q clases, concentrada en los K vecinos más cercanos
    del bin i en el espacio ab.

    pts_ab: tensor (Q, 2) con los centros de los bins en coordenadas ab reales.
    T: temperatura (más chico = distribución más picuda).
    K: cantidad de vecinos (incluyendo el propio bin).
    """

    # Aseguramos que esté en float32
    pts_ab = pts_ab.to(torch.float32)
    Q = pts_ab.shape[0]

    # Distancias euclídeas entre cada par de bins: (Q, Q)
    # dist[i, j] = distancia entre bin i y bin j en el plano ab
    dist = torch.cdist(pts_ab, pts_ab, p=2)  # (Q, Q)

    # Inicializamos puntajes con -inf
    scores = torch.full_like(dist, -float("inf"))

    # K vecinos más cercanos para cada fila (incluye al propio bin)
    knn_vals, knn_idx = dist.topk(K, dim=1, largest=False)  # (Q, K)

    # Rellenamos solo los K vecinos con -dist/T (más cerca = mayor score)
    scores.scatter_(1, knn_idx, -knn_vals / T)

    # Softmax por fila -> cada fila es una distribución de probabilidad
    soft_enc = F.softmax(scores, dim=1)  # (Q, Q)
    return soft_enc

def soft_ce_loss(logits, q_idx, soft_encoding_matrix, class_weights=None):
    """
    Cross-entropy con targets suaves (soft-encoding).

    logits: (B, Q, H, W)
    q_idx: (B, H, W)      -- índices duros por píxel
    soft_encoding_matrix: (Q, Q) -- fila i = distribución suave para clase i
    class_weights: (Q,) o None  -- rebalanceo de clases si querés
    """

    B, Q, H, W = logits.shape
    N = B * H * W  # número de píxeles en el batch

    # 1) Aplanamos logits a (N, Q)
    # (B, Q, H, W) -> (B, H, W, Q) -> (N, Q)
    logits_flat = logits.permute(0, 2, 3, 1).reshape(N, Q)

    # 2) Obtenemos los targets suaves para cada píxel: (N, Q)
    q_idx_flat   = q_idx.view(-1)                 # (N,)
    soft_targets = soft_encoding_matrix[q_idx_flat]  # (N, Q)

    # 3) Log-probabilidades predichas: (N, Q)
    log_probs = F.log_softmax(logits_flat, dim=1)

    # 4) Cross-entropy con targets suaves: - ∑ p_true * log p_pred
    loss_per_pixel = -(soft_targets * log_probs).sum(dim=1)  # (N,)

    # 5) Rebajamos o subimos pesos según frecuencia de clase (opcional)
    if class_weights is not None:
        weights = class_weights[q_idx_flat]       # (N,)
        loss_per_pixel = loss_per_pixel * weights

    # 6) Promediamos sobre todos los píxeles
    loss = loss_per_pixel.mean()
    return loss


def decode_ab_annealed(logits, pts_ab, T=0.4):
    """
    Decodifica logits de clases de color a canales a,b continuos
    usando annealed-mean con temperatura T.
    """
    # Probabilidades por bin
    probs = F.softmax(logits, dim=1)

    # Aplicar temperatura
    if T != 1.0:
        probs = probs ** (1.0 / T)
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    
    pts_ab = pts_ab.to(logits.device).float()

    ab_pred = torch.einsum('bqhw,qc->bchw', probs, pts_ab)

    return ab_pred


# ============================================
# VISUALIZACIÓN DE IMÁGENES
# ============================================

def visualize_colorization(model, dataloader, pts_ab, n_samples=5, device='cuda'):
    """
    Visualiza resultados de colorización usando el modelo de CLASIFICACIÓN.
    - logits → clase → centro de bin → canales a/b reales.
    - pts_ab: tensor (Q,2) con centros de los bins.
    """

    model.eval()

    # Obtener batch
    L_batch, q_batch = next(iter(dataloader))
    L_batch = L_batch.to(device)
    q_batch = q_batch.to(device)

    # Forward
    with torch.no_grad():
        logits = model(L_batch)    # (B, Q, H, W)

    # Convertir la grilla a numpy
    pts_ab_np = pts_ab.cpu().numpy()   # (Q,2)

    B = L_batch.shape[0]
    n_samples = min(n_samples, B)

    fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))

    for i in range(n_samples):

        # 1) Extraer L (ya normalizada)
        L_np = L_batch[i].cpu().numpy().transpose(1, 2, 0)[:, :, 0]

        # 2) Ground truth
        q_idx_true = q_batch[i].cpu().numpy()
        ab_true = pts_ab_np[q_idx_true] * 128.0

        # 3) Predicción
        q_idx_pred = torch.argmax(logits[i], dim=0).cpu().numpy()
        ab_pred = pts_ab_np[q_idx_pred] * 128.0

        # 4) Reconstrucción LAB
        H, W = L_np.shape
        lab_true = np.zeros((H, W, 3), dtype=np.float32)
        lab_pred = np.zeros((H, W, 3), dtype=np.float32)

        lab_true[:, :, 0] = L_np
        lab_true[:, :, 1:] = ab_true

        lab_pred[:, :, 0] = L_np
        lab_pred[:, :, 1:] = ab_pred

        rgb_true = np.clip(lab_to_rgb(lab_true), 0, 1)
        rgb_pred = np.clip(lab_to_rgb(lab_pred), 0, 1)

        # Error map
        error_map = np.linalg.norm(ab_true - ab_pred, axis=2)
        error_map /= error_map.max() + 1e-8

        # 5) Plots
        axes[i, 0].imshow(L_np, cmap='gray')
        axes[i, 0].set_title("Input (Grayscale)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(rgb_true)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(rgb_pred)
        axes[i, 2].set_title("Predicted")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(error_map, cmap='magma')
        axes[i, 3].set_title("Error Map")
        axes[i, 3].axis("off")

    plt.tight_layout()
    plt.show()