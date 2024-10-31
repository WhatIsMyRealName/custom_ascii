import numpy as np
import time
import sys
from PIL import Image

def progress_bar(iteration, total, timing, length=80):
    """Affiche une barre de progression.
    
    Paramètres
    --------
    - iteration : progression actuelle
    - total : nombre total d'itérations nécessaires pour arriver au bout
    - length : longueur de la barre dans le terminal
    """
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2f}% ({time.time()-timing}s)')
    sys.stdout.flush()

def read_image(image_path: str) -> np.array:
    """Lit une image et la renvoie sous forme de tableau numpy en niveaux de gris.
    - Paramètre : image_path : Chemin vers l'image à ouvrir
    - Retour : L'image sous forme de tableau numpy
    """
    print("Importation de l'image...")
    timing = time.time()
    img = Image.open(image_path).convert("L")  # Convertit en niveaux de gris # recoder cette conversion
    progress_bar(1, 1, timing)  # Fin de l'importation
    print()  # Nouvelle ligne
    return np.array(img)

def convolve(image: np.array, kernel: np.array, debug=True) -> np.array:
    """Applique une convolution de l'image.
    
    Paramètres
    --------
    - image : L'image sur laquelle réaliser la convolution
    - kernel : La matrice de convolution

    Retour
    --------
    - L'image convoluée
    """
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = np.zeros((image.shape[0] + 2 * pad_height, image.shape[1] + 2 * pad_width))
    
    # Remplir l'image pad
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            padded_image[i + pad_height, j + pad_width] = image[i, j]
    
    output = np.zeros(image.shape)
    if debug==True:
        print("Convolution de l'image...")
        timing = time.time()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Appliquer le noyau sur la région de l'image
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)  # Produit matriciel
        if debug==True: progress_bar(i + 1, image.shape[0], timing) 
    if debug==True:
        print()
    return output

def calculate_max_gradient(image: np.array) -> np.array: # ça il faut se renseigner
    """Calcule le gradient de l'image en utilisant des filtres Sobel.
    """
    # Filtres de Sobel
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
    #height, width = image.shape
    #gradient_magnitude = np.zeros_like(image, dtype=float)
    #gradient_angle = np.zeros_like(image, dtype=float)
    #for i in range(1, height - 1):
    #    for j in range(1, width - 1):
    #        gx = np.sum(image[i-1:i+2, j-1:j+2] * sobel_x)
    #        gy = np.sum(image[i-1:i+2, j-1:j+2] * sobel_y)
    #        gradient_magnitude[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    #        gradient_angle[i, j] = np.arctan2(gy, gx) * 180 / np.pi
    return [gradient_magnitude, gradient_angle]