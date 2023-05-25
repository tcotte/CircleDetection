import os


def create_directory(path):
    if not os.path.isdir(path):
        # if the directory is not present then create it.
        os.makedirs(path)