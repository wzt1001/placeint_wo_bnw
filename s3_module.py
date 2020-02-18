import os
import sys
import json
import datetime
import logging
import boto3
import time
import pandas as pd
import numpy as np

logger = logging.getLogger("__main__")

class FileSysModule(object):

    def __init__(self, bucket_name='beijing-hopson-one-hk-new'):
        """
        initiate file system module
        the default s3 bucket is etl.rifiniti.com
        there should be always configmodule already existed in the self,
        which means config
        :param bucket_name: str, bucket name
        """
        self.S3r = boto3.resource('s3')
        self.rif_bucket = self.S3r.Bucket(bucket_name)

        self.created_at = None

    def downloadfile(self, s3_path, loc_path=None):
        """
        download file from s3 and put it to specific location
        :param s3_path: str, file path from s3, e.g. clients/genentech/reference_files/path_dicts/boundary_dict.pkl
        :param loc_path: str, local path, e.g. clients/genentech/reference_files/path_dicts/boundary_dict.pkl
        """
        if loc_path is None:
            loc_path = s3_path
        self.rif_bucket.download_file(Key=s3_path, Filename=loc_path)
        logger.info("File %s is downloaded to local, %s" % (s3_path, loc_path))

    def uploadfile(self, loc_path, s3_path=None):
        """
        upload file to s3 from local
        :param s3_path: str, file path from s3, e.g. clients/genentech/reference_files/path_dicts/boundary_dict.pkl
        :param loc_path: str, local path, e.g. clients/genentech/reference_files/path_dicts/boundary_dict.pkl
        """
        if s3_path is None:
            s3_path = loc_path
        self.rif_bucket.upload_file(Filename=loc_path, Key=s3_path)
        logger.info("File %s is uploaded to S3, %s" % (loc_path, s3_path))

    def copyfileS3(self, bucket_from, bucket_to, s3_path_from, s3_path_to):
        """
        copy file from one of the Bucket to another Bucket
        :param bucket_from:
        :param bucket_to:
        :param s3_path_from:
        :param s3_path_to:
        """
        self.S3r.Object(bucket_to, s3_path_to)\
            .copy_from(CopySource=bucket_from + '/' + s3_path_from)

    def removefolder(self, folder, bucket):
        """
        remove folder from S3. Mainly used in parse_write to delete temporary folder from S3
        :param folder: str, path without bucket name
        :param bucket: str, bucket name
        """
        logger.info("delete folder from %s" % bucket)
        S3r = boto3.resource('s3')
        bucket = S3r.Bucket(bucket)

        # Not sure about this logic.
        # It seems you need to delete all the thing is the path
        delete_key_list = []
        for obj in bucket.objects.filter(Prefix=folder):
            delete_key_list.append({'Key': obj.key})
            if len(delete_key_list) > 100:
                bucket.delete_objects(Delete={'Objects': delete_key_list})
                delete_key_list = []

        if len(delete_key_list) > 0:
            bucket.delete_objects(Delete={'Objects': delete_key_list})

        logger.info("folder %s deleted." % folder)
