"""
Alex-Antoine Fortin
Tuesday December 12th 2017
Description
Makes the predictions.
"""
import os, io, imp, bson, itertools, numpy as np, pandas as pd
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm
t = imp.load_source('tools', '../code/tools.py')
model_filename_to_load = 'model-024-0.5987-1.9741.hdf5' # Str. Filename
TARGET_SIZE = (150, 150) # Tuple. Dimensions of crop size to use as input
DATASET_TO_EXTRACT = 'test.bson'
PROCESS_DATA = 'all' # Int. # of images to process, 'all': process all images
num_test_products = 1768182

categories_path = os.path.join('../input/category_names.csv')
categories_df = pd.read_csv(categories_path, index_col='category_id')

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df['category_idx'] = pd.Series(  range(len(categories_df)),
                                            index=categories_df.index )

categories_df.to_csv('../input/categories.csv')
categories_df.head()

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    return cat2idx, idx2cat

cat2idx, idx2cat = make_category_tables()

submission_df = pd.read_csv('../input/sample_submission.csv')
submission_df.head()

if PROCESS_DATA=='all':
    data = bson.decode_file_iter(open(
                '../input/{}'.format(DATASET_TO_EXTRACT), 'rb'))
elif type(PROCESS_DATA)==type(1):
    data = itertools.islice(bson.decode_file_iter(open(
                '../input/{}'.format(DATASET_TO_EXTRACT), 'rb')),PROCESS_DATA)

#================================
#Defining model & loading weights
#================================
NB_CLASS = 5270
model = t.define_model(TARGET_SIZE, NB_CLASS)
model.load_weights( os.path.join('../models/Xception-unfrozen2/',
                    model_filename_to_load))
print('Loaded the weights from model: {}'.format(model_filename_to_load))

#=======
#Predict
#=======
with tqdm(total=num_test_products) as pbar:
    """
    Performs 01 prediction  in about 1.5sec on a MacBook Pro 2015.
    Performs 25 predictions in about 1.0sec on a Tesla K80.
    Performs 21 predictions in about 1.0sec on a GTX660.
    """
    for c, d in enumerate(data):
        product_id = d['_id']
        num_imgs = len(d['imgs'])
        batch_x = np.empty((num_imgs*3, TARGET_SIZE[0], TARGET_SIZE[1], 3),
                            dtype=K.floatx())
        for i in range(num_imgs):
            bson_img = d['imgs'][i]['picture']
            # Load and preprocess the image.
            img = load_img(io.BytesIO(bson_img), target_size=(180, 180))
            x = img_to_array(img)
            x = t.preprocess_input(x)
            x = np.expand_dims(x, axis=0)
            my_crops = t.random_crop(x, TARGET_SIZE)
            for j in range(2): #3-crop
                my_crops = np.append(   my_crops, t.random_crop(x, TARGET_SIZE),
                                        axis=0)
            # Add the image to the batch.
            batch_x[i*3:(i+1)*3,:] = my_crops
        prediction_cat_idx=np.sum( model.predict(batch_x,batch_size=num_imgs*3),
                                    axis=0).argmax()
        submission_df.iloc[c]['category_id'] = idx2cat[prediction_cat_idx]
        pbar.update()

submission_df.to_csv(   '../my_submission20171212.csv.gz', compression='gzip',
                        index=False)
