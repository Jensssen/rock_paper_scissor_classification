# -*- coding: utf-8 -*-
"""
 
File:
    download.py
 
Authors: soe
Date:
    16.09.20

"""

import os
from typing import List

import boto3

from utils.filesystem_utils import create_folder

s3_resource = boto3.resource('s3')


def download_all_files(bucket: str = "rockpaperscissor903684e44d4d41a89bf150937ea98fd143614-dev",
                       save_dir_prefix: str = "/tmp"):
    """
    Downloads all folders and corresponding content from one bucket
    :param bucket: Name of the Bucket
    :param save_dir_prefix: Prefix of the folder path, where the content should be saved at
    :return:-
    """
    my_bucket = s3_resource.Bucket(bucket)
    objects = my_bucket.objects.filter(Prefix='public/')
    for obj in objects:
        path, filename = os.path.split(obj.key)
        save_dir = os.path.join(save_dir_prefix, path)
        create_folder(save_dir)
        my_bucket.download_file(obj.key, os.path.join(save_dir, filename))


def get_list_of_buckets() -> List:
    """
    Returns list of all bucket names
    :return:
        List of all bucket names
    """
    return [bucket for bucket in s3_resource.buckets.all()]

download_all_files()