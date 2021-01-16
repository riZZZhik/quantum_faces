import logging
import os

import numpy as np
from PIL import Image
from facenet_pytorch import fixed_image_standardization
from torch import from_numpy, Tensor
from torchvision import transforms


# Logging
def init_logger(log_file=None, log_level=logging.INFO, log_name=None, date_format=None):
    if date_format is None:
        date_format = {
            "format": '%(asctime)s - %(levelname)s: %(message)s',
            "datefmt": '%d-%b-%y %H:%M:%S'
        }
    else:
        assert "format" in date_format and "datefmt" in date_format, 'date_format should have "format" and "datefmt"'

    logging.basicConfig(level=log_level, **date_format)
    logger = logging.getLogger(log_name)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(*date_format.values())
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


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


def get_celeba_generator(batch_size, images_dir, labels_path, dataset_delta=None):  # TODO: Auto download CelebA
    assert os.path.exists(images_dir), \
        f"Download CelebA from https://www.kaggle.com/jessicali9530/celeba-dataset and save to {images_dir} directory"

    # Init function variables
    x, y = [], []
    x_test, y_test = [], []
    generators = {}

    # Download file paths from disk
    with open(labels_path) as f:
        for line in f:
            image_path, label = line.split()
            x.append(image_path)
            y.append(int(label))

    # Create test generator if needed
    if dataset_delta:
        x_test, y_test = x[-dataset_delta:], y[-dataset_delta:]
        x, y = x[:-dataset_delta], y[:-dataset_delta]

        def generator_test():
            for images, labels in zip(batch(x_test, batch_size), batch(y_test, batch_size)):
                x_batched = [norm_image(os.path.join(images_dir, image)) for image in images]
                yield Tensor(x_batched), from_numpy(np.array(labels))

        generators["test"] = generator_test

    # Create train generator
    def generator():
        for images, labels in zip(batch(x, batch_size), batch(y, batch_size)):
            x_batched = [norm_image(os.path.join(images_dir, image)) for image in images]
            yield Tensor(x_batched), from_numpy(np.array(labels))

    generators["train"] = generator

    # Save dataset sizes variable
    sizes = {
        "train": len(x),
        "test": len(x_test)
    }

    return generators, sizes
