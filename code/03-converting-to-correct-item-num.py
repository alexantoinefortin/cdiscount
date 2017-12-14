"""
Alex-Antoine Fortin
Wednesday, December 13th 2017
Description
Since I unpacked the image to my hard drive, I was predicting the classes in
sorted alphabetically. The script 02-prediction-gpu.py was converting idx2cat
based on their order of appearance in test.bson. This script sorts the categories
and output an ordered idx2cat. It then loads the submission.csv.gz and fixes the
columns category_id.
"""
categories_path = os.path.join("../input/category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv("../input/categories.csv")
categories_df.head()

def make_category_tables():
    cat2idx = {}
    idx2cat = {}
    cat_lst = []
    for ir in categories_df.itertuples():
        category_id = ir[0]
        category_idx = ir[4]
        cat_lst += [category_id]
        cat2idx[category_id] = category_idx
        idx2cat[category_idx] = category_id
    cat_lst = sorted(cat_lst)
    sorted_idx2cat = {x:cat_lst[x] for x in range(5270)}
    return cat2idx, idx2cat, cat_lst, sorted_idx2cat

cat2idx, idx2cat, cat_lst, sorted_idx2cat = make_category_tables()

#============================================
#Loading pandas.DataFrame with wrong item_num
#============================================
df = pd.read_csv('../submission.csv.gz', header=0)
df.columns = ['_id','category_id_wrong']
df['idx'] = df['category_id_wrong'].apply(lambda x: cat2idx.get(x, -1))
df['category_id'] = df['idx'].apply(lambda x: sorted_idx2cat.get(x, -1))
df.loc[df.category_id==-1,:]
df[['_id','category_id']].to_csv("../my_submission20171212-2.csv.gz", compression="gzip", index=False)
df = pd.read_csv('../my_submission20171212-2.csv.gz', header=0)
df.head()
