from create_dataframe import create_dataframe
import numpy as np


meta_data = create_dataframe()
def split_save():
    # 25 images per class
    value = meta_data.groupby('category_name', group_keys=False).apply(lambda x: x.sample(n=25, replace=False))
    value.sort_index(inplace=True)
    meta_data.drop(value.index, axis=0, inplace=True)

    meta_data.to_csv('train_metadata.csv', index=False)
    value.to_csv('val_metadata.csv', index=False)

def create_decoder():
    decode = {n: i for i, n in meta_data.groupby('category_name').category_number.first().iteritems()}
    np.save('decode.npy', decode)

def run_package():
    split_save()
    create_decoder()