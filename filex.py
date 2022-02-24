import cv2
import os
folder_list = os.listdir("./")
folder_list.sort()

print(folder_list)

cnt = 1

for folder in folder_list:
    if folder == "A" : continue
    if folder == "filex.py" : continue

    
    a = cv2.imread("./"+folder)
    if cnt<10:
        cv2.imwrite("./A/0"+str(cnt)+".jpg",a)
    else:
        cv2.imwrite("./A/"+str(cnt)+".jpg",a)
    cnt+=1
