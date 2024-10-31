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
        Taille de la matrice, by default 3

    Returns
    -------
    np.ndarray
        La matrice de lissage de la taille souhaitée
    """
    return np.ones(n, n) / n / n

MATRICE_DE_CONVOLUTION_DETECTION_DE_BORDS = np.array([[1, 1, 1,],
                     [1, -8, 1],
                     [1, 1, 1]])

MATRICE_DE_CONVOLUTION_CONTRASTES = np.array([[0, -1, 0],
                     [-1, 5, -1], 
                     [0, -1, 0]])

def hysteresis_thresholding(gradient_image: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    """
    Applique le suivi des contours pas hystérésis.

    Parameters
    ----------
    gradient_image : np.ndarray
        L'image dont le contenu est remplacé par les gradients en chaque point
    low_threshold : float
        Seuil bas pour les contours
    high_threshold : float
        Seuil haut

    Returns
    -------
    np.ndarray
        Image binaire avec les contours en blanc
    """
    strong_edges = (gradient_image >= high_threshold)  # Marque les bords sûrs
    weak_edges = (gradient_image >= low_threshold) & ~strong_edges  # Marque les bords faibles

    # Initialiser l'image de sortie avec les bords sûrs
    edges = np.zeros_like(gradient_image)
    edges[strong_edges] = 255

    # Propager les bords sûrs aux bords faibles adjacents
    height, width = gradient_image.shape
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if strong_edges[i, j]:  # Si c'est un bord fort, propager aux voisins
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if weak_edges[i + di, j + dj]:  # Si le voisin est un bord faible
                            edges[i + di, j + dj] = 255  # Inclure ce voisin dans les contours

    return edges

def calculate_thresholds(image: np.ndarray, low_percentile=95, high_percentile=98) -> tuple:
    """
    Calcule les valeurs de l'image correspondant au centiles entrés en paramètre.

    Parameters
    ----------
    image : np.ndarray
        L'image à traier
    low_percentile : int, optional
        Le premier centile, bas, by default 95
    high_percentile : int, optional
        Le second centile, haut, by default 98

    Returns
    -------
    tuple
        (centile_bas, centile_haut)
    """
    low_threshold = np.percentile(image, low_percentile)
    high_threshold = np.percentile(image, high_percentile)
    return low_threshold, high_threshold

def map_character(gradiant: list, average_gray_level: float, color=True) -> str:
    """
    Fais correspondre un caractère ASCII à un gradiant.

    Parameters
    ----------
    gradiant : list
        Le grandiant à convertir sous la forme [norme, argument]
    average_gray_level : float
        Niveau moyen de gris, utilisé si le gradiant est trop faible et si color=True
    color : bool, optional
        Permet de convertir aussi les parties sans bords (unies donc) en fonction de la couleur moyenne, by default True

    Returns
    -------
    str
        Le caractère choisi.
    """
    if gradiant[0] > 10:
        symbols = ['-', '/', '|', '\\']
        index = int((gradiant[1] + 22.5) // 45) % 4
        return symbols[index]
    else:
        if color:
            grayscale = '@%#*+=-:. '
            return grayscale[int(average_gray_level*(len(grayscale)-1)/255)]
        else:
            return "."

def image_to_ascii(image: str, or_img: str, block_size=20, height_reduction_factor=0.6, color=True) -> str:
    """
    Convertit l'image en ASCII. Fonction principale.

    Parameters
    ----------
    image : str
        L'image des bords
    or_img : str
        L'image originale (utilisée si color=True, pour avoir les niveaux de gris en plus des bords)
    block_size : int, optional
        Nombre de pixels qui vont être représentés par 1 caractère ASCII, by default 20.
    height_reduction_factor : float, optional
        Facteur d'"applatissement" de l'image pour compenser la déformation dues aux caractères ASCII, plus haut que larges, by default 0.6.
        NOTE : ce coefficient est aussi appliqué à block_size pour rester cohérant. Basez-vous donc plutôt sur la largeur de l'image pour déterminer
            le paramètre block_size
    color : bool, optional
        Pour colorier l'image, by default True. Metre à False pour ne convertir que les bords.

    Returns
    -------
    str
        L'image convertie
    """
    height, width = image.shape
    ascii_art = ""
    print("Conversion de l'image en ASCII...")
    yspeed = int(block_size/height_reduction_factor)
    notexact = (height % yspeed)!=0
    timing = time.time()
    for i in range(0, height, yspeed):
        for j in range(0, width, block_size):
            block = image[i:min(i + yspeed, height), j:min(j + block_size, width)]
            or_block = or_img[i:min(i + yspeed, height), j:min(j + block_size, width)]
            magnitude, angle = calculate_max_gradient(block)
            average_gray_level = np.mean(or_block)
            char = map_character((magnitude, angle), average_gray_level, color)
            ascii_art += char
        ascii_art += "\n"
        progress_bar(i // yspeed, height // yspeed + 1*notexact, timing)
    progress_bar(1, 1, timing) # juste pour avoir un affichage cohérant
    print() 
    return ascii_art

def display_ascii_art(ascii_art: str) -> None:
    """
    Affiche l'image dans la console.

    Parameters
    ----------
    ascii_art : str
        L'image ASCII à afficher
    """
    print(ascii_art)

def save_ascii_art(ascii_art: str, name="result") -> None:
    """
    Enregistre l'image ASCII.

    Parameters
    ----------
    ascii_art : str
        L'image ASCII à enregistrer
    name : str, optional
        Chemin vers le fichier d'enregistrement, by default "result". Sera créé si besoin, écrasé si nécessaire.
        NOTE : Attention, sur VSCode, il faut mettre le chemin par rapport au dossier ouvert, pas pas rapport au fichier. Sur IDLE,
            le chemin relatif suffit.
    """
    with open(f'{name}.txt', 'w') as f:
        f.write(ascii_art)

# from scipy.ndimage import convolve as convolveCheat

def main(image_path: str, block_size=20, color=True):
    """
    Permet de tester le code ou de servir d'exemple en utilisant les fonctions précédentes.

    Parameters
    ----------
    image_path : str
        Chemin vers l'image à convertir
        NOTE : Attention, sur VSCode, il faut mettre le chemin par rapport au dossier ouvert, pas pas rapport au fichier. Sur IDLE,
            le chemin relatif suffit.
    block_size : int, optional
        Nombre de pixels qui vont être représentés par 1 caractère ASCII, by default 20.
    color : bool, optional
        Pour colorier l'image, by default True. Metre à False pour ne convertir que les bords.
    """
    image = read_image(image_path)
    smoothed_image = np.clip(convolve(image, average_kernel()), 0, 255)
    contrasted_image = np.clip(convolve(smoothed_image, MATRICE_DE_CONVOLUTION_CONTRASTES), 0, 255)
    test = np.clip(convolve(contrasted_image, MATRICE_DE_CONVOLUTION_DETECTION_DE_BORDS), 0, 255)

    low_threshold, high_threshold = calculate_thresholds(test)
    final_edges = hysteresis_thresholding(test, low_threshold, high_threshold)
    # sans hystérésis :
    #print("Moyenne des normes des gradiants des bords détectés : ", np.mean(np.abs(test)[test != 0]))
    #result_image = np.where(np.abs(test) > 75, 255, 0) # remplacer 75 par la moyenne calculée ci-dessus, mais généralement pas trop mal pour un premier test
    #result_image = result_image.astype(np.uint8)
    ascii_art = image_to_ascii(final_edges, image, block_size, color=color)
    display_ascii_art(ascii_art)
    save_ascii_art(ascii_art)

    # Affichage de débogage :
    fig = plt.figure(figsize=(10, 10)) 
    fig.add_subplot(2, 2, 1) 
    plt.imshow(image, cmap="gray") 
    plt.axis('off') 
    plt.title("Image") 

    fig.add_subplot(2, 2, 2) 
    plt.imshow(smoothed_image, cmap="gray") 
    plt.axis('off') 
    plt.title("Image lissée")

    fig.add_subplot(2, 2, 3) 
    plt.imshow(contrasted_image, cmap="gray") 
    plt.axis('off') 
    plt.title("Image contrastée") 

    fig.add_subplot(2, 2, 4) 
    plt.imshow(final_edges, cmap="gray") 
    plt.axis('off') 
    plt.title("Bords")

    plt.show()

# main('image.jpg', block_size=8 ,color=True)
if __name__ == '__main__':
    main('image.jpg', block_size=8 ,color=True)