# import cv2
# cap = cv2.VideoCapture(0)
# while True:
#     ret, frame = cap.read()
#     cv2.imshow("Face-Recognition", frame)
#     key = cv2.waitKey(1)
#     if key == 27:
#         break
# cv2.destroyAllWindows()
# cap.release()


import numpy as np

arr1 = np.zeros((3,5))
arr2 = np.zeros((5,6))
arr3 = np.asarray([arr1,arr2])
print(arr3)