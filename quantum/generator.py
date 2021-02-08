import os
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image
from keras.utils import Sequence, to_categorical
from loguru import logger

from .image_preparation import ImagePreparation


class Generator(Sequence):  # TODO: Function to split images in Train and Val generators
    """Keras generator class to CelebA dataset"""
    def __init__(self, batch_size, image_shape, images_dir, labels_path, label_max_filter=None,
                 face_shape_predict_model=None, crop_type=1):
        """Init main class variables.

        :param batch_size: Batch size
        :type batch_size: int
        :param image_shape: Image shape
        :type image_shape: list or tuple
        :param images_dir: Path to images dir
        :type images_dir: str
        :param labels_path: Path to labels file
        :type labels_path: str
        :param label_max_filter: Max label id to filter images
        :type label_max_filter: int
        :param face_shape_predict_model: Path to dlib face_shape_predict_model
        :type face_shape_predict_model: str
        :param crop_type: Crop type id (0 - dont crop, 1 - crop and align face, 2 - images with 68-points coordinates)
        :type crop_type: int
        """

        assert len(image_shape) == 3, 'Image shape should have 3 dimensions'

        # Initialize class variables
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.images_dir = images_dir
        self.labels_path = labels_path
        self.crop_type = crop_type

        self.images, self.labels = [], []
        self.image_preparation = ImagePreparation(face_shape_predict_model)

        # Download dataset if needed
        if not os.path.exists(images_dir):
            self.download_dataset()

        # Get labels
        with open(labels_path) as f:
            for line in f:
                image_path, label = line.split()
                label = int(label)
                if not label_max_filter or label < label_max_filter:
                    self.images.append(os.path.join(images_dir, image_path))
                    self.labels.append(label)

        self.num_classes = max(self.labels) + 1
        self.labels = to_categorical(self.labels)

        logger.info(f"Found {len(self.images)} images with {self.num_classes} classes")

    def __len__(self):
        """Get dataset length.

        :return: Dataset length
        """

        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        """Get batch of images.

        :param index: Current index
        :type index: int
        :return: Batch of normalized images
        """

        images = []
        ready_images_paths = []

        while len(images) < self.batch_size:  # TODO: Optimize this script
            images_paths = self.images[index * self.batch_size:(index + 1) * self.batch_size]

            for image_path in images_paths:
                try:
                    if image_path not in ready_images_paths:  # TODO: Check part with skipping image
                        images.append(self.norm_image_from_disk(image_path))
                        ready_images_paths.append(image_path)
                except ValueError as e:  # TODO: Check why always only one image skipped?
                    logger.warning(f"Removing this image from dataset because of error: {str(e)}")
                    err_index = self.images.index(image_path)
                    del self.images[err_index]
                    np.delete(self.labels, err_index)

        labels = np.array(self.labels[index * self.batch_size:(index + 1) * self.batch_size])
        return np.array(images), labels

    def download_dataset(self):
        """"Download dataset if required"""
        if not os.path.exists(self.images_dir):
            logger.info(f"No CelebA dataset found in {self.images_dir}, downloading from server")

            if os.path.exists("tmp"):
                del_tmp = False
            else:
                os.mkdir("tmp")
                del_tmp = True

            os.system("mkdir ~/.kaggle")
            os.system("cp dataset/kaggle.json ~/.kaggle/")
            os.system("kaggle datasets download -d jessicali9530/celeba-dataset -p tmp")

            with zipfile.ZipFile("tmp/celeba-dataset.zip", 'r') as zip_ref:
                zip_ref.extractall("tmp/")
            shutil.move("tmp/img_align_celeba/img_align_celeba", self.images_dir)
            if del_tmp:
                os.system("rm -rf tmp")

    def norm_image_from_disk(self, image_path):
        """Download and normalize image from disk.

        :param image_path: Path to image
        :type image_path: str
        :return: Normalized image array
        """

        assert self.crop_type or self.image_preparation.face_detector, \
            "You didn't given face_shape_predict_model path to class"

        image = Image.open(image_path)

        if self.crop_type == 0:
            return self.image_preparation.resize_image(image, self.image_shape, image_path)
        else:
            faces = self.image_preparation.crop_faces(image)
            if not faces:
                raise ValueError(f"Cant find face on {image_path} image")

            if self.crop_type == 2:
                aligned_face = self.image_preparation.get_aligned_frame(np.array(faces[0]))
                if aligned_face:
                    shape = aligned_face.shape
                    coordinates = self.image_preparation.get_face_shape_coordinates(aligned_face)
                    if coordinates:
                        cropped_image = np.zeros(shape[:2], np.int8)
                        size = int(shape[0] / 60)
                        for x, y in coordinates[-1]:
                            if shape[0] > y >= 0 and shape[1] > x >= 0:
                                cropped_image = cv2.circle(cropped_image, (x, y), size, 255, -1)

                        cropped_image = Image.fromarray(cropped_image, "L")
                        return self.image_preparation.resize_image(cropped_image, self.image_shape, image_path)
                    else:
                        raise ValueError(f"Cant find face coordinates on the {image_path} image")
                else:
                    raise ValueError(f"Cant find face coordinates on the {image_path} image")

            else:
                return self.image_preparation.resize_image(faces[0], self.image_shape, image_path)
