import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = [8, 8]

def plot_hist(gray_image, thresh):
    """Función para mostrar el histograma de una imagen en escala de grises."""
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    plt.plot(np.array([thresh, thresh]), np.array([0, np.max(hist)]), color='red', linestyle='--')
    plt.plot(hist)
    plt.show()

def get_brightness(gray_image):
    """Calcula el brillo de una imagen en escala de grises."""
    return round(np.sum(gray_image) / gray_image.size)

def get_contrast(gray_image):
    """Calcula el contraste de una imagen en escala de grises."""
    return round(np.max(gray_image) - np.min(gray_image))

def bottle_inspection(input_image, image_name):
    """Realiza la inspección de una botella."""
    numero_inspecciones = 4
    resultados = [None] * numero_inspecciones
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    out_image = input_image.copy()

    # Aquí va el código para las inspecciones individuales
    # ...

    if all(resultados):
        cv2.rectangle(out_image, (0, 0), out_image.shape[1::-1], (0, 255, 0), 50)
    else:
        cv2.rectangle(out_image, (0, 0), out_image.shape[1::-1], (255, 0, 0), 50)

    plt.title(image_name)
    plt.imshow(out_image)
    plt.show()

def main():
    images_path = 'data/bottle_inspection'
    files = os.listdir(images_path)

    input_images = []
    for file in files:
        image = cv2.cvtColor(cv2.imread(os.path.join(images_path, file)), cv2.COLOR_BGR2RGB)
        input_images.append({'image': image, 'name': file})

    for img_info in input_images:
        bottle_inspection(img_info['image'], img_info['name'])

if __name__ == "__main__":
    main()
