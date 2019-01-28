from get_data import run_data_checker
from clean_data import clean_data
from create_decoder import  create_decoder
from split_save_data import split_save

#in this order
# 1. Check if data exists. If not get data
# 2. Clean data
# 3. Create the decoder
# 4. Create the training.csv files

run_data_checker()
clean_data()
create_decoder()
split_save()