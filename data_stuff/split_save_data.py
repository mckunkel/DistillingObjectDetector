from create_dataframe import create_dataframe

def split_save():
    # 25 images per class
    meta_data = create_dataframe()
    value = meta_data.groupby('category_name', group_keys=False).apply(lambda x: x.sample(n=25, replace=False))
    value.sort_index(inplace=True)
    meta_data.drop(value.index, axis=0, inplace=True)

    meta_data.to_csv('train_metadata.csv', index=False)
    value.to_csv('val_metadata.csv', index=False)
