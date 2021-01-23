import shutil
import zipfile

from loguru import logger
import os

import numpy as np
from PIL import Image
from facenet_pytorch import fixed_image_standardization
from torch import from_numpy, Tensor
from torchvision import transforms

# Images
trans = transforms.Compose([
    np.float32,
    fixed_image_standardization
])


def norm_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))
    img = trans(img)
    img = np.moveaxis(img, -1, 0)

    return img


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        data = iterable[ndx:min(ndx + n, l)]
        delta = n - len(data)
        if delta:
            data += iterable[:delta]
        yield data


def get_celeba_generator(batch_size, images_dir, labels_path, dataset_delta=None, label_max_filter=None):
    # Auto download CelebA
    if not os.path.exists(images_dir):
        logger.info(f"No CelebA dataset found in {images_dir}, downloading dataset from server")
        if os.path.exists("tmp"):
            del_tmp = False
        else:
            os.mkdir("tmp")
            del_tmp = True

        os.system("mkdir ~/.kaggle")
        os.system("cp dataset/kaggle.json ~/.kaggle/")

        os.system("kaggle datasets download -d jessicali9530/celeba-dataset -p tmp")

        with zipfile.ZipFile("tmp/celeba-dataset", 'r') as zip_ref:
            zip_ref.extractall("tmp/")

        shutil.move("tmp/img_align_celeba/img_align_celeba", images_dir)
        if del_tmp:
            os.remove("tmp")

    # Init function variables
    x, y = [], []
    x_test, y_test = [], []
    generators = {}

    # Download file paths from disk
    with open(labels_path) as f:
        for line in f:
            image_path, label = line.split()
            label = int(label)
            if label_max_filter and label < label_max_filter:
                x.append(image_path)
                y.append(label)

    num_classes = max(y) + 1

    # Create test generator if needed
    if dataset_delta:
        x_test, y_test = x[-dataset_delta:], y[-dataset_delta:]
        x, y = x[:-dataset_delta], y[:-dataset_delta]

        # noinspection DuplicatedCode
        def generator_test():
            for images, labels in zip(batch(x_test, batch_size), batch(y_test, batch_size)):
                try:
                    x_batched = [norm_image(os.path.join(images_dir, image)) for image in images]
                    yield Tensor(x_batched), from_numpy(np.array(labels))
                except FileNotFoundError:
                    logger.warning(f"One from {images} images not found, skipping this batch")
                    continue

        generators["val"] = generator_test

    # Create train generator
    # noinspection DuplicatedCode
    def generator():
        for images, labels in zip(batch(x, batch_size), batch(y, batch_size)):
            try:
                x_batched = [norm_image(os.path.join(images_dir, image)) for image in images]
                yield Tensor(x_batched), from_numpy(np.array(labels))
            except FileNotFoundError:
                logger.warning(f"One from {images} images not found, skipping this batch")
                continue

    generators["train"] = generator

    # Save dataset sizes variable
    sizes = {
        "train": len(x),
        "val": len(x_test)
    }

    return generators, sizes, num_classes
