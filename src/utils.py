"""
Most codes from https://github.com/carpedm20/DCGAN-tensorflow
"""

import os


def checkfolder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

