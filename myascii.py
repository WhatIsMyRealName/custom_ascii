import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import time

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

def average_kernel() -> np.array:
    """Crée un noyau de moyenne 3x3 pour le lissage."""
    return np.ones((3, 3)) / 9.0

def convolve(image: np.array, kernel: np.array) -> np.array: # est-ce que c'est vraiment utilse ça ?
    """Applique une convolution de l'image.
    
    Paramètres
    --------
    - image : L'image sur laquelle réaliser la convolution
    - kernel : La matrice de convolution

    Retour
    --------
    - L'image convoluée (réduite de quelques pixels sur chaque bord)
    """
    kernel_height, kernel_width = kernel.shape
    pad_height, pad_width = kernel_height // 2, kernel_width // 2
    padded_image = np.zeros((image.shape[0] + 2 * pad_height, image.shape[1] + 2 * pad_width))
    
    # Remplir l'image pad
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            padded_image[i + pad_height, j + pad_width] = image[i, j]
    
    output = np.zeros(image.shape)
    print("Convolution de l'image...")
    timing = time.time()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Appliquer le noyau sur la région de l'image
            region = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)  # Produit matriciel
        progress_bar(i + 1, image.shape[0], timing) 
    print() 
    return output

def edge_detection(image: np.array) -> np.array:
    """Détecte les contours basés sur la différence de luminosité avec les voisins.
    - Paramètre : image : L'image à analyser
    - Retour : Nouvelle image sous format np contenant les contours de l'image en blancs, le reste en noir
    """
    height, width = image.shape
    output = np.zeros_like(image)  # Image de sortie noire
    print("Détection des contours...")
    timing = time.time()
    for i in range(1, height-1):
        for j in range(1, width - 1):
            # Calculer la différence de luminosité avec les voisins
            neighbors = [
                image[i - 1, j],  # haut
                image[i + 1, j],  # bas
                image[i, j - 1],  # gauche
                image[i, j + 1]   # droite
            ]
            # Calculer la différence maximale
            max_difference = max(neighbors) - min(neighbors)

            # Si la différence est significative (seuil ajustable)
            if max_difference > 20:  # Ajustez ce seuil selon vos besoins
                output[i, j] = 255  # Pixel blanc
            else:
                output[i, j] = 0    # Pixel noir
        progress_bar(i, height - 2, timing) 
    print()  # Nouvelle ligne
    return output

def calculate_gradient(image: np.array) -> np.array: # ça il faut se renseigner
    """Calcule le gradient de l'image en utilisant des filtres Sobel.
    - Paramètre : image : L'image à analyser
    - Retour : Matrice des gradiants en chaque points de l'image
    """
    # Filtres de Sobel
    sobel_x = np.array([[1, 0, -1],
                         [2, 0, -2],
                         [1, 0, -1]])

    sobel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])

    height, width = image.shape
    gradient_magnitude = np.zeros_like(image, dtype=float)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            gx = np.sum(image[i-1:i+2, j-1:j+2] * sobel_x)
            gy = np.sum(image[i-1:i+2, j-1:j+2] * sobel_y)
            gradient_magnitude[i, j] = np.sqrt(gx ** 2 + gy ** 2)
    return gradient_magnitude

def map_character(magnitude: float) -> str:
    """Mappe la magnitude à un caractère ASCII.
    - Paramètre : magnitude : La magnitude ("pente") à convertir en ASCII
    - Retour : le caractère ASCII correspondant    
    """
    # les bords
    if magnitude > 100:
        return '|'
    elif magnitude > 70:
        return '-'
    elif magnitude > 50:
        return '_'
    elif magnitude > 30:
        return '/'
    elif magnitude > 10:
        return '\\'
    else:
        return '.' # pas un bord => l'étape d'après c'est de convertir en fonction du niveau gris
    #vous voyez les trucs unis c'est des .

    # bon faut que j'y ailles les gars
    # merci
    # je vous tiens au courant 

def image_to_ascii(image: str, block_size=8, height_reduction_factor=0.5) -> str: # il marche bizarrement ce facteur
    """Convertit l'image en ASCII art en utilisant des blocs.
    
    Paramètres
    --------
    - image : L'image à convertir
    - block_size :  Largeur en pixels du bloc qui sera représenté par 1 caractère ASCII. 
    - height_reduction_factor : Facteur de réduction de la hauteur pour compenser la déformation dues aux caractères ASCII

    Retour
    --------
    - ascii_art : L'image convertie
    """
    height, width = image.shape
    ascii_art = ""
    print("Conversion de l'image en ASCII...")
    yspeed = int(block_size/height_reduction_factor)
    exact = (height % yspeed)!=0
    timing = time.time()
    for i in range(0, height, yspeed):
        for j in range(0, width, block_size):
            # Découper le bloc # S'assurer de ne pas dépasser les limites de l'image
            block = image[i:min(i + yspeed, height), j:min(j + block_size, width)]
            # Calculer la magnitude des gradients
            gradient_magnitude = np.mean(calculate_gradient(block))  # Utiliser la moyenne pour le bloc
            # Mapper à un caractère
            char = map_character(gradient_magnitude)
            ascii_art += char
        ascii_art += "\n"
        progress_bar(i // yspeed, height // yspeed + 1*exact, timing)
    progress_bar(1, 1, timing) # juste pour avoir un affichage cohérant
    print() 
    return ascii_art

def display_ascii_art(ascii_art: str) -> None:
    """
    Affiche l'ASCII art dans la console.
    - Paramètre : ascii_art : ascii art sous forme de chaîne de caractères
    """
    print(ascii_art)

def save_ascii_art(ascii_art: str, name="result") -> None:
    """
    Enregistre l'ascii art dans le fichier {name}.txt

    Paramètres
    --------
    - ascii_art : ascii art sous forme de chaîne de caractères
    - name : nom du fichier texte où enregistrer le résultat sour forme de chaîne de caractères
    """
    with open(f'{name}.txt', 'w') as f:
        f.write(ascii_art)

image_path = "ascii art /chartest5.png"
image = read_image(image_path)
smoothed_image = convolve(image, average_kernel())
result_image = edge_detection(smoothed_image)
ascii_art = image_to_ascii(result_image)
display_ascii_art(ascii_art)
save_ascii_art(ascii_art)

# create figure 
fig = plt.figure(figsize=(10, 7)) 
fig.add_subplot(1, 3, 1) 
plt.imshow(image, cmap="gray") 
plt.axis('off') 
plt.title("Image") 

fig.add_subplot(1, 3, 2) 
plt.imshow(smoothed_image, cmap="gray") 
plt.axis('off') 
plt.title("Image lissée") 

fig.add_subplot(1, 3, 3) 
plt.imshow(result_image, cmap="gray") 
plt.axis('off') 
plt.title("Bords")
plt.show()