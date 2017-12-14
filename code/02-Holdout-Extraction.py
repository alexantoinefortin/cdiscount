#!/usr/local/bin/python3
"""
Alex-Antoine Fortin
December 07, 2017
Description
Extracts images from test.bson as a 180x180x3 JPG in ../input/holdout/holdout/<_id>.jgp
"""
import numpy as np, pandas as pd, os
"""
The bson files for this competition contain a list of dictionaries,
one dictionary per product. Each dictionary contains a product id (key: _id),
the category id of the product (key: category_id), and between 1-4 images,
stored in a list (key: imgs). Each image list contains a single dictionary per image,
which uses the format: {'picture': b'...binary string...'}.
"""
import io, bson, multiprocessing as mp
import itertools
from PIL import Image
from time import time
from random import random
#%matplotlib inline
startTime = time()

#==============
#Distributed document reading and saving
#==============
NCORE =  8
RESIZE_PICTURES = False
SIZE = (180,180)
DATASET_TO_EXTRACT = 'test.bson'
PROCESS_DATA = 'all' # integer: num of images to process, 'all': process all of them

if not os.path.isdir('../input/holdout/holdout/'): # Create top folder if needed
    os.makedirs('../input/holdout/holdout/')

prod_to_category = mp.Manager().dict()
def process(q, iolock):
    while True:
        d = q.get()
        if d is None:
            break
        product_id = d['_id']
        #category_id = d['category_id']
        for e, pic in enumerate(d['imgs']):
            picture = Image.open(io.BytesIO(pic['picture']))
            path_to_save = '../input/holdout/holdout'
            if RESIZE_PICTURES:
                picture.resize(SIZE, Image.LANCZOS).save(os.path.join(path_to_save,str(product_id)+'.jpg'))
            else:
                picture.save(os.path.join(path_to_save,str(product_id)+'.jpg'))

q = mp.Queue(maxsize=NCORE)
iolock = mp.Lock()
pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))

# process files
if PROCESS_DATA=='all':
    data = bson.decode_file_iter(open('../input/{}'.format(DATASET_TO_EXTRACT), 'rb'))
elif type(PROCESS_DATA)==type(1):
    data = itertools.islice(bson.decode_file_iter(open('../input/{}'.format(DATASET_TO_EXTRACT), 'rb')),PROCESS_DATA)

for c, d in enumerate(data):
    q.put(d)  # blocks until q below its max size

# tell workers we're done
for _ in range(NCORE):
    q.put(None)
pool.close()
pool.join()

#===================
# Count number of file created and report processing time
#===================
path = '../input/holdout/holdout'
def compute_stats(path):
    nb_img = sum([len(files) for r, d, files in os.walk(path)])
    return nb_img

nb_img = compute_stats(path)
print("created {} files in ../input/holdout/holdout/ in {} seconds.".format(nb_img, int(time()-startTime)))
