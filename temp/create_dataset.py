import os

from imutils import paths

path_img = r"C:\Users\tristan_cotte\Pictures\Dataset_legio\SM\23_05\Labelme\2nd_labelme"

if __name__ == "__main__":
    for i in list(paths.list_images(path_img)):
        photo_name = i.split("\\")[-1][:-4]
        open(os.path.join("labels", photo_name + ".txt"), 'a').close()
