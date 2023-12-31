import matplotlib.pyplot as plt
import matplotlib as mpl
import cv2
import numpy as np
import os

mpl.rcParams['figure.figsize'] = [8, 8]

def background_substraction(input_image, name, bkg_image):
    output_image = cv2.absdiff(cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY),
                               cv2.cvtColor(bkg_image, cv2.COLOR_RGB2GRAY))
    T, output_image = cv2.threshold(output_image, 0, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Guarda la imagen en lugar de mostrarla
    cv2.imwrite(f'output_{name}.png', output_image)
    return output_image

def find_cars(original_image, background_image, min_area, max_area):
    bkg_image = background_image.copy()
    out_image = original_image.copy()
    kernel = np.ones((5,5),np.uint8)
    bkg_image = cv2.morphologyEx(bkg_image, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((35,35),np.uint8)
    bkg_image = cv2.morphologyEx(bkg_image, cv2.MORPH_CLOSE, kernel)
    contours = cv2.findContours(bkg_image, cv2.RETR_LIST, 
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
    xcnts = []
    for contour in contours:
        if min_area < cv2.contourArea(contour) <= max_area:
            xcnts.append(contour)
            cv2.drawContours(out_image, contour, -1, (0, 255, 0), 3)
        else:
            cv2.drawContours(out_image, contour, -1, (255, 0, 0), 3)
    return out_image, bkg_image, len(xcnts)

images_path = 'data/highway'
files = sorted(os.listdir(images_path))
input_images = []

for file in files:
    input_image = cv2.cvtColor(cv2.imread(os.path.join(images_path, file)), cv2.COLOR_RGB2BGR)
    input_images.append({'image': input_image, 'name': file})

background_images = []
start_image = 1

for i in range(start_image, len(input_images) - 2):
    bkg_image = background_substraction(input_images[i]['image'], input_images[i]['name'], 
                                        input_images[0]['image'])
    background_images.append({'image': bkg_image, 'name': file})

area_min = 15000
area_max = 50000
output_images = []

for i in range(len(background_images)):
    out_image, bkg_image, n_cars = find_cars(input_images[i+1]['image'], 
                                             background_images[i]['image'],
                                             area_min, area_max)
    output_images.append(out_image)
    # Guarda la imagen en lugar de mostrarla
    cv2.imwrite(f'detected_cars_{i}.png', out_image)

start_image = 0

for i in range(start_image, len(output_images)):
    logo_resized = cv2.resize(input_images[7]['image'], output_images[i].shape[1::-1], interpolation=cv2.INTER_AREA)
    out_image = cv2.addWeighted(output_images[i], 0.9, logo_resized, 0.1, 0)
    # Guarda la imagen en lugar de mostrarla
    cv2.imwrite(f'watermarked_{i}.png', out_image)
