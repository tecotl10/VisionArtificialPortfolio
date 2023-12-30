import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np
import os

def load_image(path):
    """Carga una imagen y la convierte a formato BGR."""
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image is not None else None

def apply_perspective_transform(original, original_points, destination_points, output_size):
    """Aplica transformación de perspectiva a la imagen."""
    M = cv2.getPerspectiveTransform(original_points, destination_points)
    return cv2.warpPerspective(original, M, (output_size, output_size))

def affine_transform_method(original, dst, original_points, destination_points, output_size):
    """Aplica una transformación afín a la imagen."""
    M = cv2.getAffineTransform(original_points[:3], destination_points[:3])
    return cv2.warpAffine(original, M, (output_size, output_size))

def main():
    # Configuración
    mpl.rcParams['figure.figsize'] = [8, 8]
    rocket_path = 'data/Rocket.jpg'
    tans_path = 'data/tans'
    
    # Cargar y mostrar imagen original
    rocket = load_image(rocket_path)
    if rocket is None:
        print("Error al cargar la imagen.")
        return
    
    plt.imshow(rocket)
    plt.show()

    # Transformación de la imagen original
    original_points = np.float32([[650, 1100], [2425, 1050], [400, 2800], [2700, 2850]])
    destination_points = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
    transformed_rocket = apply_perspective_transform(rocket, original_points, destination_points, 600)

    # Mostrar la imagen transformada
    plt.subplot(121), plt.imshow(rocket), plt.title('Input')
    plt.subplot(122), plt.imshow(transformed_rocket), plt.title('Output')
    plt.show()

    # Procesar imágenes de tangram
    files = os.listdir(tans_path)
    output_images = []
    for file in files:
        image = load_image(os.path.join(tans_path, file))
        if image is not None:
            # Aplicar transformaciones aquí (ejemplo genérico)
            transformed_image = affine_transform_method(image, transformed_rocket, original_points, destination_points, 600)
            output_images.append(transformed_image)

    # Combinar imágenes transformadas
    result_image = output_images[0]
    for image in output_images[1:]:
        result_image = cv2.add(result_image, image)

    # Mostrar imagen final
    plt.imshow(result_image)
    plt.show()

if __name__ == "__main__":
    main()
