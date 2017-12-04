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

