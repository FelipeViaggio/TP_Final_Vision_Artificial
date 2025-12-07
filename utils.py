from skimage import color

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
