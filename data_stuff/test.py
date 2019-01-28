import os
import constants as c
import matplotlib.pyplot as plt

sub_dirs = list(os.walk(c.dir_train))[1:]
dir_path = sub_dirs[0][0];
dir_name = sub_dirs[0][0].split('/')[-1]
files = sub_dirs[0][2][1]
print(dir_path)
print(dir_name)
print(files)
image = plt.imread(os.path.join(dir_path, files))
print(image.shape)

# for file_name in files:
#     print(file_name)
#     image = plt.imread(os.path.join(dir_path, file_name))
# for i in range(3):
#     print(sub_dirs[i])
# for dir_path, _, files in sub_dirs:
#     dir_name = dir_path.split('/')[-1]
#     print(dir_name)
#     print(len(files))
