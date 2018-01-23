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
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
WORKER_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_worker_support.h5")

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
    NUM_CLASSES = 1 + 3  # COCO has 80 classes

############################################################
#  Dataset
############################################################

class WorkerSupportDataset(utils.Dataset):
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
        for i in range(len(imagesList)):
            im = Image.open(os.path.join(image_dir, imagesList[i]))
            width, height = im.size
            self.add_image(
                "worker_support", image_id=i,
                path=os.path.join(image_dir, imagesList[i]),
                json_path=os.path.join(json_dir, imagesList[i][:-4]+'__labels.json'),
                width=width, height=height
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


############################################################
#  COCO Evaluation
############################################################




############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on WorkerSupport Dataset.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' on WorkerSupport Dataset")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/worker_support/",
                        help='Directory of the WorkerSupport dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'worker_support'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (defaults=500)')
    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = WorkerSupportConfig()
    else:
        class InferenceConfig(WorkerSupportConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to save/load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
    elif args.model.lower() == "worker_support":
        # Find last trained weights
        model_path = WORKER_MODEL_PATH
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        model_path = args.model

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                        exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = WorkerSupportDataset()
        dataset_train.load_worker_support(args.dataset, "train")
        dataset_train.prepare()

        # Validation dataset
        dataset_val = WorkerSupportDataset()
        dataset_val.load_worker_support(args.dataset, "test")
        dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')
        
        print("Saving weights ", model_path)
        model.keras_model.save_weights(model_path)

    elif args.command == "evaluate":
        pass
        # Load weights
        print("Loading weights ", model_path)
        model.load_weights(model_path, by_name=True)
        # Validation dataset
        # dataset_val = CocoDataset()
        # coco = dataset_val.load_coco(args.dataset, "minival", return_coco=True)
        # dataset_val.prepare()
        # print("Running COCO evaluation on {} images.".format(args.limit))
        # evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))