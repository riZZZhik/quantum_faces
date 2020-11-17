import os
from glob import glob

import face_recognition
import matplotlib.pyplot as plt
from cv2 import circle
import numpy as np
from PIL import Image
from resizeimage import resizeimage


def crop_faces(image):
    image_array = np.array(image)
    locations = face_recognition.face_locations(image_array)

    face_images = []
    for location in locations:
        top, right, bottom, left = location
        rect = (left, top, right, bottom)
        face_images.append(image.crop(rect))
    return face_images


def image_normalization(image, w, h):
    image = resizeimage.resize_cover(image, [w, h])

    try:
        image = np.array([[image.getpixel((x, y))[0] for x in range(w)] for y in range(h)])
    except TypeError:
        image = np.array([[image.getpixel((x, y)) for x in range(w)] for y in range(h)])

    # 2-dim data convert to 1-dim array
    image = image.flatten()

    # change type
    image = image.astype('float64')

    # Normalization(0~pi/2)
    image /= 255.0
    generated_image = np.arcsin(image)

    return generated_image


def get_paths_to_images(images_path):
    assert type(images_path) in (list, tuple, str), "images_paths type should be list or tuple or str"

    if type(images_path) == str:
        assert os.path.isdir(images_path), "Path with images does not exists"
        images_path = glob(images_path + "/*.jpg") + glob('quantum_faces/faces/*.jpeg') + glob(images_path + "/*.png")
    else:
        assert all(os.path.isfile(p) for p in images_path)

    return images_path


def norm_images_from_disk(images_path: (list, tuple, str), resize_size: (list, tuple), plt_show=True):
    assert type(resize_size) in (list, tuple) and len(resize_size) == 2, \
        "resize_size type should be list or tuple and length == 2"

    images_path = get_paths_to_images(images_path)

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


def norm_face_images_from_disk(images_path: (list, tuple, str), resize_size: (list, tuple), plt_show=True):
    assert type(resize_size) in (list, tuple) and len(resize_size) == 2, \
        "resize_size type should be list or tuple and length == 2"

    images_path = get_paths_to_images(images_path)

    images = []
    face_images = []
    norm_images = []

    for path in images_path:
        images.append(Image.open(path))
        faces = crop_faces(images[-1])
        if faces:
            face_images.append(*faces)
            for face in faces:
                norm_images.append(image_normalization(face.convert('LA'), *resize_size))

    if plt_show:
        for face in face_images:
            plt.imshow(face)
            plt.show()

    return images, face_images, norm_images


def norm_face_landmarks_images_from_disk(images_path: (list, tuple, str), resize_size: (list, tuple), plt_show=True):
    assert type(resize_size) in (list, tuple) and len(resize_size) == 2, \
        "resize_size type should be list or tuple and length == 2"

    images_path = get_paths_to_images(images_path)

    images = []
    face_images = []
    landmarks = []
    landmarks_images = []
    norm_images = []

    for path in images_path:
        images.append(Image.open(path))
        faces = crop_faces(images[-1])
        if faces:
            face_images.append(*faces)
            for face in faces:
                face = np.array(face)
                shape = face.shape

                landmarks.append([])
                for i in face_recognition.face_landmarks(face)[0].values():
                    landmarks[-1] += i

                landmarks_images.append(np.zeros(shape[:2], np.int8))
                radius = int(shape[0] / 60)
                for x, y in landmarks[-1]:
                    if shape[0] > y >= 0 and shape[1] > x >= 0:
                        landmarks_images[-1] = circle(landmarks_images[-1], (x, y), radius, 255, -1)  # FIXME: Different face marks size

                landmarks_images[-1] = Image.fromarray(landmarks_images[-1], "L")
                norm_images.append(image_normalization(landmarks_images[-1], *resize_size))

    if plt_show:
        for face, marks in zip(face_images, landmarks_images):
            plt.imshow(face)
            plt.show()
            plt.imshow(marks)
            plt.show()

    return images, face_images, landmarks, landmarks_images, norm_images
