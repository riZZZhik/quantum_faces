import os
import shutil
import zipfile

import numpy as np
from PIL import Image
from keras.utils import Sequence, to_categorical
from loguru import logger

from .image_preparation import ImagePreparation


class Generator(Sequence):  # TODO: Function to split images in Train and Val generators
    def __init__(self, batch_size, image_shape, images_dir, labels_path, label_max_filter=None,
                 face_shape_predict_model=None):
        assert len(image_shape) == 3, 'Image shape should have 3 dimensions'

        # Initialize class variables
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.images_dir = images_dir
        self.labels_path = labels_path

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
        return len(self.images) // self.batch_size

    def __getitem__(self, index):
        images_paths = self.images[index * self.batch_size:(index + 1) * self.batch_size]
        images = self.image_preparation.norm_images_from_disk(images_paths, 1, self.image_shape)

        labels = np.array(self.labels[index * self.batch_size:(index + 1) * self.batch_size])
        return images, labels

    def download_dataset(self):
        logger.info(f"No CelebA dataset found in {self.images_dir}, downloading dataset from server")

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

    def norm_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize(self.image_shape[:2])
        return np.array(img)
