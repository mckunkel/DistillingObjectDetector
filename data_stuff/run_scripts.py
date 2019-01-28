from get_data import run_data_checker
from clean_data import clean_data
from split_save_data import  run_package

#in this order
# 1. Check if data exists. If not get data
# 2. Clean data
# 3. Create the decoder
# 4. Create the training.csv files

print('Checking for data')
run_data_checker()
print('Cleaning for data')
clean_data()
print('Running decoder and split save decoder')
run_package()