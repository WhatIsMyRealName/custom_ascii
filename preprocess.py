import numpy as np
import sys
import time
from PIL import Image
import warnings
try:
    from scipy.ndimage import convolve as scipy_convolve
    scipy_available = True
except ImportError:
    scipy_available = False
    warnings.warn("""\033[93m\nScipy n'est pas disponible. Le paramètre fast est automatiquement corrigé en False.
Exécutez 'pip install scipy' '!pip install scipy' ou 'py -m pip install scipy' pour pouvoir utiliser fast=True.\033[0m""", category=UserWarning)

def progress_bar(iteration: int, total: int, timing: float, length=80):
    """
    Affiche une barre de progression pour permettre à l'utilisateur de savoir où il en est.

    Parameters
    ----------
    iteration : int
        Numéro de la tache qu'on est en train de faire
    total : int
        Nombre total d'itérations pour compléter la tâche
    timing : float
        Heure de début
    length : int, optional
        Longueur de la barre dans la console, by default 80
    """
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2f}% ({time.time()-timing}s)')
    sys.stdout.flush()

def read_image(image_path: str) -> np.ndarray:
    """
    Ouvre l'image et la convertit en niveaux de gris et en np.ndarray.

    Parameters
    ----------
    image_path : str
        Chemin vers l'image.
        NOTE : Attention, sur VSCode, il faut mettre le chemin par rapport au dossier ouvert, pas pas rapport au fichier. Sur IDLE,
            le chemin relatif suffit.

    Returns
    -------
    np.array
        L'image convertie en niveaux de gris
    """
    print("Importation de l'image...")
    timing = time.time()
    img = Image.open(image_path).convert("L")  # Convertit en niveaux de gris
    progress_bar(1, 1, timing) 
    print()
    return np.array(img)

def convolve(image: np.ndarray, kernel: np.ndarray, debug=True, fast=True) -> np.ndarray:
    """
    Applique une convolution sur l'image avec un calcul optimisé si `fast=True`.

    Parameters
    ----------
    image : np.ndarray
        L'image avec 1 canal sous-entendu et 2 dimensions (i.e. image.shape == (height, width)).
    kernel : np.ndarray
        La matrice de convolution à appliquer.
    debug : bool, optional
        Affiche une barre de progression si True, par défaut True.
    fast : bool, optional
        Utilise une convolution rapide avec scipy.ndimage.convolve si True, par défaut True.

    Returns
    -------
    np.ndarray
        L'image convoluée.
    """
    if fast and not scipy_available:
        fast = False  # Bascule automatiquement si scipy est absent
    
    if fast:
        image = image.astype(np.float32)
        return scipy_convolve(image, kernel, mode='constant', cval=0.0)
    
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = np.zeros((image.shape[0] + 2 * pad_height, image.shape[1] + 2 * pad_width))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            padded_image[i + pad_height, j + pad_width] = image[i, j]
    
    output = np.zeros(image.shape)
    if debug==True:
        print("Convolution de l'image...")
        timing = time.time()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)  # "Produit" matriciel
        if debug==True: progress_bar(i + 1, image.shape[0], timing) 
    if debug==True:
        print()
    return output

def calculate_max_gradient(image: np.ndarray) -> list:
    """
    NOTE : cette version devrait être renommée calculate_mean_gradient.

    Calcule la moyenne des gradiants dans l'image de bords donnée.

    Parameters
    ----------
    image : np.ndarray
        L'image à traiter. A priori une petite partie d'une grande image

    Returns
    -------
    list
        Les moyennes des normes et arguments des gradiants de l'image
    """
    sobel_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    x = convolve(image, sobel_x, False)
    y = convolve(image, sobel_y, False)
    gradient_magnitude = np.mean(np.sqrt(x**2 + y**2))
    gradient_angle = np.mean(np.arctan2(y, x) * 180 / np.pi)
    return [gradient_magnitude, gradient_angle]