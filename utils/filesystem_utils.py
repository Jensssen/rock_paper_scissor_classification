# -*- coding: utf-8 -*-
"""
 
File:
    filesystem_utils.py
 
Authors: soe
Date:
    17.09.20

"""

import os


def create_folder(path: str):
    """
    Creates a new folder under path if it does not already exist
    :param
        path: name/path to folder
    :return:
        -
    """
    if not os.path.exists(path):
        os.makedirs(path)
