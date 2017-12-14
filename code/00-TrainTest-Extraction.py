#!/usr/local/bin/python3
"""
Alex-Antoine Fortin
November 23rd 2017
Description
Extracts img from train.bson as a 180x180x3 JPG in ../input/category_id/_id.jgp
"""
import numpy as np, pandas as pd, os
# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
"""
The bson files for this competition contain a list of dictionaries,
one dictionary per product. Each dictionary contains a product id (key: _id),
the category id of the product (key: category_id), and between 1-4 images,
stored in a list (key: imgs). Each image list contains a single dictionary per
image, which uses the format: {'picture': b'...binary string...'}.
"""
import io, bson, itertools, multiprocessing as mp
from PIL import Image
from time import time
from random import random
PCT_TO_SAVE_AS_TEST = 0.01 # Float.
startTime = time()
#==============
#Distributed document reading and saving
#==============
NCORE =  8
RESIZE_PICTURES = False
SIZE = (180,180)
DATASET_TO_EXTRACT = 'train.bson'
PROCESS_DATA = 'all' # Int. # of images to process, 'all': process all of them

if not os.path.isdir('../input/train/'): # Create top folder if needed
    os.makedirs('../input/train/')
    os.makedirs('../input/test/')

prod_to_category = mp.Manager().dict()
def process(q, iolock):
    while True:
        d = q.get()
        if d is None:
            break
        product_id = d['_id']
        category_id = d['category_id']
        for e, pic in enumerate(d['imgs']):
            picture = Image.open(io.BytesIO(pic['picture']))
            if random()<PCT_TO_SAVE_AS_TEST:
                path_to_save = '../input/test/{}'.format(category_id)
            else:
                path_to_save = '../input/train/{}'.format(category_id)
            if not os.path.isdir(path_to_save):
                os.mkdir(path_to_save)
            if RESIZE_PICTURES:
                picture.resize(SIZE, Image.LANCZOS).save(
                os.path.join(path_to_save,str(product_id)+'_'+str(e)+'.jpg')
                )
            else:
                picture.save(
                os.path.join(path_to_save,str(product_id)+'_'+str(e)+'.jpg')
                )

q = mp.Queue(maxsize=NCORE)
iolock = mp.Lock()
pool = mp.Pool(NCORE, initializer=process, initargs=(q, iolock))

# process files
if PROCESS_DATA=='all':
    data = bson.decode_file_iter(
            open('../input/{}'.format(DATASET_TO_EXTRACT), 'rb'))
elif type(PROCESS_DATA)==type(1):
    data = itertools.islice(bson.decode_file_iter(
            open('../input/{}'.format(DATASET_TO_EXTRACT), 'rb')),PROCESS_DATA)

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
path_train = '../input/train/'
path_test = '../input/test/'
def compute_stats(path):
    nb_img = sum([len(files) for r, d, files in os.walk(path)])
    categories = [name for name in os.listdir(path)
                                    if os.path.isdir(os.path.join(path, name))]
    cat_cnts = []
    for files in categories:
        cat_cnts += [len(os.listdir(os.path.join(path,str(files))))]
    nb_categories = len(categories)
    return(nb_img, cat_cnts, nb_categories)

nb_img, cat_cnts, nb_categories = compute_stats(path_train)
print('created {} files and {} classes in ../input/train/ in {} seconds.'\
                        .format(nb_img, nb_categories, int(time()-startTime)))
nb_img, cat_cnts, nb_categories = compute_stats(path_test)
print('created {} files and {} classes in ../input/test/ in {} seconds.'\
                        .format(nb_img, nb_categories, int(time()-startTime)))

#=======================================================================
#Creating empty dir in case we missed some classes in the validation set
#=======================================================================

d = '../input/train'
categories = [o for o in os.listdir(d) if os.path.isdir(os.path.join(d,o))]

dd = '../input/test'
for classes in categories:
    if not os.path.isdir(os.path.join(dd, classes)):
        os.makedirs(os.path.join(dd, classes))
