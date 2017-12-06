"""
Mask R-CNN
Configurations and data loading code for unsafe behavior with supports and workers detection

Copyright (c) 2017 HUST.
Licensed under the MIT License (see LICENSE for details)
Written by Shuangjie Xu

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 worker_support.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 worker_support.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 worker_support.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 worker_support.py train --dataset=/path/to/coco/ --model=last

    # Run Worker and Support evaluatoin on the last model you trained
    python3 worker_support.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np
from PIL import Image, ImageDraw
import json as js

from config import Config
import utils
import model as modellib

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_worker_support.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class WorkerSupportConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "worker_support"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

############################################################
#  Dataset
############################################################

class CocoDataset(utils.Dataset):
    def load_worker_support(self, dataset_dir, subset, class_ids=None,
                  class_map=None):
        """Load a subset of the worker_support dataset.
        dataset_dir: The root directory of the worker_support dataset.
        subset: What to load (train, val, minival, val35k)
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        """
        # Add classes
        self.add_class("worker_support", 1, "steel_support")
        self.add_class("worker_support", 2, "cement_support")
        self.add_class("worker_support", 3, "person")
        # Path
        image_dir = os.path.join(dataset_dir, "train/images" if subset == "train"
                                 else "test/images")
        json_dir = os.path.join(dataset_dir, "train/labels" if subset == "train"
                                 else "test/labels")
        imagesList = os.listdir(image_dir)
        imageNum = len(imagesList)
        for i in range(imageNum):
            # TODO: add width and height
            self.add_image(
                "worker_support", image_id=i,
                path=os.path.join(image_dir, imagesList[i]),
                json_path=os.path.join(json_dir, imagesList[i][:-4]+'.json'),
                width=100, height=100
            )

    def image_reference(self, image_id):
        """Return the worker_support data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "worker_support":
            return info["worker_support"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        json_path = info['json_path']
        with open(json_path, 'r') as f:
            content = js.load(f)
        count = len(content['labels'])
        nx, ny = info['width'], info['height']
        mask = np.zeros([nx, ny, count], dtype=np.uint8)
        class_ids = np.zeros([count], dtype=np.int32)
        for i, (annotation) in enumerate(content['labels']):
            # label class
            class_name = annotation['label_class']
            vertices = annotation['vertices']
            poly = []
            for vertice in vertices:
                poly.append((float(vertice['x']), float(vertice['y'])))
            img = Image.new("L", [nx, ny], 0)
            ImageDraw.Draw(img).polygon(poly, outline=1, fill=1)
            mask[:, :, i:i+1] = np.array(img)
            class_ids[i:i+1] = self.class_names.index(class_name)
        return mask, class_ids
            
        
        