
from __future__ import print_function

import os
#import numpy as np
import pandas as pd
import tarfile
import urllib.request
import zipfile
from glob import glob


class Download():
    """Init Download of NYCflight"""
    def __init__(self, data_dir):
        self.data_dir

class Flights(Download):
    """"Download csv"""
    def __init__(self, url, row_numbers):
        self.url = url
        self.data_dir = 'data'
        self.row_numbers = int(row_numbers)
        flights_raw = os.path.join(self.data_dir, 'nycflights.tar.gz')
        flightdir = os.path.join(self.data_dir, 'nycflights')
        jsondir = os.path.join(self.data_dir, 'flightjson')

        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)

        if not os.path.exists(flights_raw):
            print("- Downloading NYC Flights dataset... ", end='', flush=True)
            url = self.url
            urllib.request.urlretrieve(url, flights_raw)
            print("done", flush=True)

        if not os.path.exists(flightdir):
            print("- Extracting flight data... ", end='', flush=True)
            tar_path = os.path.join(self.data_dir, 'nycflights.tar.gz')
            with tarfile.open(tar_path, mode='r:gz') as flights:
                flights.extractall('data/')
            print("done", flush=True)

        if not os.path.exists(jsondir):
            print("- Creating json data... ", end='', flush=True)
            os.mkdir(jsondir)
            for path in glob(os.path.join(self.data_dir, 'nycflights', '*.csv')):
                prefix = os.path.splitext(os.path.basename(path))[0]
                # Just take the first 10000 rows for the demo
                df = pd.read_csv(path).iloc[:self.row_numbers]
                df.to_json(os.path.join(self.data_dir, 'flightjson', prefix + '.json'),
                        orient='records', lines=True)
            print("done", flush=True)

        print("** Finished! **")