import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from PIL import Image
from resizeimage import resizeimage


def image_normalization(image, w, h):
    image = resizeimage.resize_cover(image, [w, h])
    image = np.array([[image.getpixel((x, y))[0] for x in range(w)] for y in range(h)])

    # 2-dim data convert to 1-dim array
    image = image.flatten()

    # change type
    image = image.astype('float64')

    # Normalization(0~pi/2)
    image /= 255.0
    generated_image = np.arcsin(image)

    return generated_image


def norm_images_from_disk(images_path: (list, tuple, str), resize_size: (list, tuple), plt_show=True):
    assert type(resize_size) in (list, tuple) and len(resize_size) == 2

    if type(images_path) == str:
        assert os.path.exists(images_path)
        images_path = glob(images_path + "/*.jpg") + glob(images_path + "/*.png")
        print(images_path)
        return
    else:
        assert type(images_path) in (list, tuple)


    images = []
    norm_images = []

    for path in images_path:
        images.append(Image.open(path).convert('LA'))
        norm_images.append(image_normalization(images[-1], *resize_size))

    if plt_show:
        for image in images:
            plt.imshow(image)
            plt.show()

    return images, norm_images
