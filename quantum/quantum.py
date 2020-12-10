import logging

from .utils import init_logger


class Quantum():
    def __init__(self, log_file="logs.log", log_level=logging.DEBUG):
        # Init logger
        self.logger = init_logger(log_file, log_level, __name__)
        self.logger.info("Initializing Quantum class")

        self.dataset = None
        self.dataset_sizes = None

    def get_face_dataset(self):
        """Get faces dataset from sklearn fetch_lfw_people and save it to self.dataset"""
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import fetch_lfw_people

        self.logger.info("Initializing sklearn fetch_lfw_people dataset")
        # Load data
        lfw_dataset = fetch_lfw_people(min_faces_per_person=100)

        # Save train, test and val datasets to self.dataset
        _, h, w = lfw_dataset.images.shape
        x = lfw_dataset.images[:-40]
        y = lfw_dataset.target[:-40]
        n_components = len(lfw_dataset.target_names)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        x_val, y_val = lfw_dataset.images[-40:], lfw_dataset.target[-40:]

        self.dataset = {
            "train": [x_train, y_train],
            "test": [x_test, y_test],
            "val": [x_val, y_val],
            "n_components": n_components
        }
        self.dataset_sizes = {x: len(self.dataset[x]) for x in self.dataset}
