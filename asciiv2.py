import numpy as np
import matplotlib.pyplot as plt
import time
from preprocess import progress_bar, read_image, convolve, calculate_max_gradient

def average_kernel(n=3) -> np.ndarray:
    """
    Crée une matrice de lissage de taille spécifiée.

    Parameters
    ----------
    n : int, optional
        Taille de la matrice de lissage, par défaut 3.

    Returns
    -------
    np.ndarray
        La matrice de lissage de taille `n x n`.
    """
    return np.ones((n, n)) / (n**2)

MATRICE_DE_CONVOLUTION_DETECTION_DE_BORDS = np.array([[1, 1, 1],
                                                      [1, -8, 1],
                                                      [1, 1, 1]])

MATRICE_DE_CONVOLUTION_CONTRASTES = np.array([[0, -1, 0],
                                              [-1, 5, -1], 
                                              [0, -1, 0]])

SOBEL_X = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])

SOBEL_Y = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])

def calculate_thresholds(image: np.ndarray, low_percentile=95, high_percentile=98) -> tuple:
    """
    Calcule les seuils bas et haut pour une image en fonction des percentiles.

    Parameters
    ----------
    image : np.ndarray
        Image à traiter.
    low_percentile : int, optional
        Centile bas, par défaut 95.
    high_percentile : int, optional
        Centile haut, par défaut 98.

    Returns
    -------
    tuple
        (low_threshold, high_threshold), seuils pour les contours faibles et forts.
    """
    low_threshold = np.percentile(image[np.where(image > 0)], low_percentile)
    high_threshold = np.percentile(image[np.where(image > 0)], high_percentile)
    return low_threshold, high_threshold

def hysteresis_thresholding(gradient_image: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """
    Applique le suivi des contours par hystérésis.

    Parameters
    ----------
    gradient_image : np.ndarray
        L'image de gradient.
    low_threshold : float
        Seuil bas pour les contours faibles.
    high_threshold : float
        Seuil haut pour les contours forts.

    Returns
    -------
    np.ndarray
        Image binaire avec les contours marqués en blanc (255).
    """
    strong_edges = (gradient_image >= high_threshold)
    weak_edges = (gradient_image >= low_threshold) & ~strong_edges
    edges = np.zeros_like(gradient_image)
    edges[strong_edges] = 255

    height, width = gradient_image.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if strong_edges[i, j]:
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if weak_edges[i + di, j + dj]:
                            edges[i + di, j + dj] = 255

    return edges

def detect_edges_and_angles(image: np.ndarray, low_thresh: float|None = None, high_thresh: float|None = None) -> tuple:    
    Gx = convolve(image, SOBEL_X, fast=True)
    Gy = convolve(image, SOBEL_Y, fast=True)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    
    if low_thresh is None or high_thresh is None:
        low_thresh, high_thresh = calculate_thresholds(magnitude)

    edges = hysteresis_thresholding(magnitude, low_thresh, high_thresh) # type: ignore

    angles = np.zeros_like(magnitude)
    angles[edges > 0] = np.arctan2(Gy[edges > 0], Gx[edges > 0]) * 180 / np.pi

    return edges, angles

def map_character(gradient: tuple, average_gray_level: float|None = None) -> str:
    """
    Associe un caractère ASCII à un gradient.

    Parameters
    ----------
    gradient : tuple
        Gradient sous la forme (norme, anlge).
    average_gray_level : float, optional
        Niveau moyen de gris, utilisé si le gradient est faible.

    Returns
    -------
    str
        Caractère ASCII correspondant.
    """
    
    if gradient[0] > 10:
        symbols = ['-', '\\', '|', '/']
        index = int((gradient[1] - 90 + 25) // 45) % 4
        return symbols[index]
    if average_gray_level is not None:
        grayscale = '@%#*+=-:. '
        return grayscale[int(average_gray_level * (len(grayscale) - 1) / 255)]
    else:
        return " "
    
def image_to_ascii(image: np.ndarray, or_img: np.ndarray|None = None, block_size=20, height_reduction_factor=0.6) -> str:
    """
    Convertit l'image en ASCII.

    Parameters
    ----------
    image : np.ndarray
        Image des bords.
    or_img : np.ndarray, optional
        Image originale utilisée pour calculer les niveaux de gris si le gradient est faible, par défaut None
    block_size : int, optional
        Taille des blocs de pixels pour un caractère ASCII, par défaut 20.
    height_reduction_factor : float, optional
        Facteur d'aplatissement de l'image pour correspondre à la largeur des caractères ASCII, par défaut 0.6.

    Returns
    -------
    str
        Représentation ASCII de l'image.
    """
    height, width = image.shape
    ascii_art = "" # "_" + "_" * (width//block_size) + "_" + "\n" + "|"
    print("Conversion de l'image en ASCII...")
    yspeed = int(block_size / height_reduction_factor)
    notexact = (height % yspeed) != 0
    timing = time.time()
    for i in range(0, height, yspeed):
        for j in range(0, width, block_size):
            block = image[i:min(i + yspeed, height), j:min(j + block_size, width)]
            if or_img is not None:
                magnitude, angle = calculate_max_gradient(block)
                or_block = or_img[i:min(i + yspeed, height), j:min(j + block_size, width)]
                average_gray_level = float(np.mean(or_block)) # float != float[Any]
                char = map_character((magnitude, angle), average_gray_level)
            else:
                magnitude, angle = calculate_max_gradient(block)
                char = map_character((magnitude, angle))
            ascii_art += char
        ascii_art += "\n" # "|" + "\n" + "|"
        progress_bar(i // yspeed, height // yspeed + 1 * notexact, timing)
    ascii_art += "" # "_" * (width//block_size) + "|"
    progress_bar(1, 1, timing)
    print()
    return ascii_art

def display_ascii_art(ascii_art: str) -> None:
    """
    Affiche l'art ASCII dans la console.

    TODO : autoriser les `StrOrBytesPath | IO[bytes]` pour lire un fichier
    
    Parameters
    ----------
    ascii_art : str
        Représentation ASCII de l'image.
    """
    print(ascii_art)

def save_ascii_art(ascii_art: str, name="result") -> None:
    """
    Enregistre l'art ASCII dans un fichier texte.

    Parameters
    ----------
    ascii_art : str
        Représentation ASCII de l'image.
    name : str, optional
        Nom de fichier, par défaut "result".
    """
    with open(f'{name}.txt', 'w') as f:
        f.write(ascii_art)

def main(image_path: str, block_size: int = 20, color: int = True, fast: bool = True, height_reduction_factor: float = 0.6):
    """
    Convertit une image en ASCII et affiche les étapes de traitement.

    Parameters
    ----------
    image_path : str
        Chemin vers l'image à convertir.
    block_size : int, optional
        Taille des blocs de pixels pour un caractère ASCII, par défaut 20.
    color : bool, optional
        Appliquer le niveau de gris moyen, par défaut True.
    fast : bool, optional
        Mode rapide de convolution.
    height_reduction_factor : float, optional
        Facteur d'aplatissement de l'image.
    """
    
    image = read_image(image_path)
    smoothed_image = np.clip(convolve(image, average_kernel(), fast=fast), 0, 255)
    contrasted_image = np.clip(convolve(smoothed_image, MATRICE_DE_CONVOLUTION_CONTRASTES, fast=fast), 0, 255)
    
    # ici, utiliser Sobel une première fois
    
    x = convolve(image, SOBEL_X, False)
    y = convolve(image, SOBEL_Y, False)
    test = np.clip(np.sqrt(x**2 + y**2), 0, 255)
    
    # test = np.clip(convolve(image, MATRICE_DE_CONVOLUTION_DETECTION_DE_BORDS, fast=fast), 0, 255)
    # test = edge_detection(image)
    
    low_threshold, high_threshold = calculate_thresholds(test)
    print(low_threshold, high_threshold)
    final_edges = hysteresis_thresholding(test, low_threshold, high_threshold)

    s = ""
    yspeed = int(block_size / height_reduction_factor)
    for k in range(len(final_edges) // yspeed +1):
        for i in range(k*yspeed, min(len(final_edges), (k+1)*yspeed)):
            for j in range(len(final_edges[i]) // block_size +1):
                s += " ".join([str(x)[:2] for x in final_edges[i][j*block_size:(j+1)*block_size]]) + "|"
            s += '\n'
        s += "---"*(len(final_edges[0])) + '\n'
    
    # final_edges = detect_edges_and_angles(image) # tuple (edges, angles)
    ascii_art = image_to_ascii(final_edges, image if color else None, block_size, height_reduction_factor)
    
    save_ascii_art(s, "test")
    save_ascii_art(ascii_art)
    
    """
    print("\n\n")
    display_ascii_art(ascii_art)
    
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(2, 3, 1)
    plt.imshow(image, cmap="gray")
    #plt.axis('off')
    plt.title("Image")

    fig.add_subplot(2, 3, 2)
    plt.imshow(x, cmap="gray")
    #plt.axis('off')
    plt.title("x")

    fig.add_subplot(2, 3, 3)
    plt.imshow(y, cmap="gray")
    #plt.axis('off')
    plt.title("y")

    fig.add_subplot(2, 3, 4)
    plt.imshow(test, cmap="gray")
    #plt.axis('off')
    plt.title("Bords bruts")

    carre = np.clip(convolve(final_edges, MATRICE_DE_CONVOLUTION_DETECTION_DE_BORDS, fast=fast), 0, 255)
    fig.add_subplot(2, 3, 5)
    plt.imshow(final_edges, cmap="gray")
    #plt.axis('off')
    plt.title("Bords")

    plt.show()
    """

def main_sans_debug(image_path: str, block_size: int = 20, color: int = True, fast: bool = True, height_reduction_factor: float = 0.6):
    """
    Convertit une image en ASCII rapidement.    

    Parameters
    ----------
    image_path : str
        Chemin vers l'image à convertir.
    block_size : int, optional
        Taille des blocs de pixels pour un caractère ASCII, par défaut 20.
    color : bool, optional
        Appliquer le niveau de gris moyen, par défaut True.
    fast : bool, optional
        Mode rapide de convolution.
    height_reduction_factor : float, optional
        Facteur d'aplatissement de l'image.
    """
    image = read_image(image_path)
    x = convolve(image, SOBEL_X, False)
    y = convolve(image, SOBEL_Y, False)
    test = np.clip(np.sqrt(x**2 + y**2), 0, 255)
    low_threshold, high_threshold = calculate_thresholds(test)
    final_edges = hysteresis_thresholding(test, low_threshold, high_threshold)
    ascii_art = image_to_ascii(final_edges, image if color else None, block_size, height_reduction_factor)
    save_ascii_art(ascii_art)

if __name__ == '__main__':
    main('image.jpg', block_size=8, color=True, fast=True, height_reduction_factor=.4)