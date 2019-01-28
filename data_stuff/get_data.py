import os, tarfile
import download_data as dl
import constants as c


def checkDirIfExists():
    return os.path.isdir(c.dir_train)


def checkDirEmpty():
    len_of_dir = 0
    if checkDirIfExists():
        len_of_dir = len(list(os.walk(c.dir_train))[1:])
    return len_of_dir


def checkIfTarExists():
    return os.path.exists(c.file_train)


def untar(fname):
    if (fname.endswith("tar")):
        tar = tarfile.open(fname, "r:")
        tar.extractall()
        tar.close()
        print ("Extracted in Current Directory")


if checkDirIfExists() and checkDirEmpty()>200:
    print('Files already in folder {}'.format(c.dir_train))
elif checkIfTarExists():
    print('The tar file is already downloaded. Extracting tar file')
    untar(c.file_train)
else:
    print('No data or .tar file exists')
    print('Downloading the .tar file')
    dl.getData()
    untar(c.file_train)

# if __name__ == '__main__':
#     print(checkDirIfExists())
#     print(checkDirEmpty())
#     print(checkIfTarExists())


