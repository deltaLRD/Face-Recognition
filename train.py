import cv2
import os
for path, dir_list, file_list in os.walk("./data"):
    print(path)
    print(dir_list)
    print(file_list)
    break
