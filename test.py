import cv2
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

def imread(filename, flags=cv2.IMREAD_COLOR, dtype=np.uint8): 
    try: 
        n = np.fromfile(filename, dtype) 
        img = cv2.imdecode(n, flags) 
        return img 
    except Exception as e: 
        print(e) 
        return None

def imwrite(filename, img, params=None): 
    try: 
        ext = os.path.splitext(filename)[1] 
        result, n = cv2.imencode(ext, img, params) 
        if result: 
            with open(filename, mode='w+b') as f: 
                n.tofile(f) 
            return True 
        else: 
            return False 
    except Exception as e: 
        print(e) 
        return False


def datapreprocessing(folderlist, PATH):
    x_test = []
    tmplist= [] 
    for folder in folderlist:
        file_list = os.listdir(PATH + folder)
        file_list.sort()
        for filename in file_list:
            filepath = PATH + folder + '/' + filename
            image = imread(filepath)
       
         
            image = cv2.resize(image, (128,128))
            image.tolist()
            image = image / 255.
            x_test.append(image)
            tmplist.append(filepath)

            '''
            a = cv2.imread(filepath)
        
            a1 = cv2.flip(a, 1) 
            a2 = cv2.flip(a, 0) 
            a3 = cv2.flip(a, 1) 
            a3 = cv2.flip(a3, 0) 

            cv2.imwrite(PATH + folder + '/'+filename.split(".")[0]+"_h.jpg",a1)
            cv2.imwrite(PATH + folder + '/'+filename.split(".")[0]+"_v.jpg",a2)
            cv2.imwrite(PATH + folder + '/'+filename.split(".")[0]+"_hv.jpg",a3)
            '''

    return x_test, file_list, tmplist

def calCropScore(cropsize, x, pred):
    start = 0
    absval = abs(x-pred)
    absval = absval[3:125, 3:125]
    #absval1 = np.array(absval[:,:,0]).flatten().tolist()
    #absval2 = np.array(absval[:,:,1]).flatten().tolist()
    #absval3 = np.array(absval[:,:,2]).flatten().tolist()
    #absval = absval1 + absval2 + absval3
    #absval = sorted(absval, reverse = True)
    absval = np.mean(absval,axis=2)
    absval = np.array(absval).flatten().tolist()
    absval = sorted(absval, reverse = True)
    #score = np.mean(absval)
    #print(absval)
    #absval[absval<0.2] = 0.5
    score = np.sum(absval[start:cropsize*cropsize])/(cropsize*cropsize-start)

    return score

def calScore(x_test, prediction, classv,name):
    imageShape = (128, 128,3)
    reconCrop = 10
    #boostCrop = 4
    #param = 0
    tempList = []
    #sharpening_mask = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    

    for i in range(0,x_test.shape[0]):
        '''
        patch_x = []
        patch_p = []
        for j in range(4):
            for k in range(4):
                patch_x.append(x_test[i][j*32:(j+1)*32,k*32:(k+1)*32])
                patch_p.append(prediction[i][j*32:(j+1)*32,k*32:(k+1)*32])

        temp = []

        for j in range(16):
            print(patch_x[j].shape)
            temp.append(calCropScore(reconCrop, patch_x[j], patch_p[j]))

        print(temp)
        reconScore = max(temp)
        print(reconScore)
        '''
        reconScore = calCropScore(reconCrop, x_test[i], prediction[i])
        result = np.mean(abs(prediction[i]-x_test[i]),axis=2)
        #result = abs(prediction[i]-x_test[i])
        '''
        if classv == "normal":
            cv2.imwrite("./reconstruction_subtract/"+classv+"/"+name[i], result*255)
        else:
            fname = name[i].split("/")
            #print(fname)
            
            cv2.imwrite("./reconstruction_subtract/"+classv+"/"+fname[3]+fname[4],result *255)
        
        
        boost_test=np.array(x_test[i]) 
        boost_test = np.reshape(boost_test, imageShape)
        boost_test = cv2.filter2D(boost_test, -1, sharpening_mask)

        boost_prediction=np.array(prediction[i]) 
        boost_prediction = np.reshape(boost_prediction,imageShape)
        boost_prediction = cv2.filter2D(boost_prediction, -1, sharpening_mask)

        boostScore = calCropScore(boostCrop, boost_test, boost_prediction)
        '''
        tempList.append(reconScore)
        
    return tempList
    

    
if __name__ == "__main__" :
    
    NORMAL_PATH = './newtest/normal/'
    ABNORMAL_PATH = './newtest/abnormal/'

    folder_list = os.listdir(NORMAL_PATH)
    folder_list.sort()

    abfolder_list = os.listdir(ABNORMAL_PATH)
    abfolder_list.sort()

    x_test_normal, normalfile_list, _ = datapreprocessing(folder_list, NORMAL_PATH)
    x_test_abnormal, abnormalfile_list, tmplist = datapreprocessing(abfolder_list, ABNORMAL_PATH)
    
    x_test_normal = np.array(x_test_normal)
    x_test_abnormal = np.array(x_test_abnormal)

    model = tf.keras.models.load_model('solarModel2')
    
    predictions_normal = model.predict(x_test_normal)
    predictions_abnormal = model.predict(x_test_abnormal)
    NormalList = calScore(x_test_normal, predictions_normal,"normal",normalfile_list)
    AbnormalList = calScore(x_test_abnormal, predictions_abnormal,"abnormal",tmplist)

    x_data = NormalList+AbnormalList
    x_data = (x_data - min(x_data)) / (max(x_data) - min(x_data))

    normal_true = [0 for i in range(int(x_test_normal.shape[0]))]
    abnormal_true = [1 for i in range(int(x_test_abnormal.shape[0]))]
    print(normal_true, abnormal_true)
    fpr, tpr, threshold = roc_curve(normal_true+abnormal_true, x_data)
    optimal_index = np.argmax(tpr - fpr) 
    optimal_threshold = threshold[optimal_index]
    
    sumTrue = normal_true + abnormal_true
    sumPred= [0 if i<optimal_threshold else 1 for i in x_data]

    print(x_data[:int(x_test_normal.shape[0])], sumPred[:int(x_test_normal.shape[0])])
    print(x_data[int(x_test_normal.shape[0]):], sumPred[int(x_test_normal.shape[0]):])
     
    cnf_matrix = confusion_matrix(sumTrue, sumPred)
  
    print(classification_report(sumTrue, sumPred,digits=3))
    print(cnf_matrix)
    print(optimal_threshold)
   
    plt.plot(fpr, tpr, linewidth=2,label="ROC curve (area = %0.3f)" % roc_auc_score(sumTrue, sumPred))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend(loc="lower right")   
    plt.show() 

    plt.hist(x_data[:int(x_test_normal.shape[0])], alpha=0.7, bins=100, label='Normal')
    plt.hist(x_data[int(x_test_normal.shape[0]):], alpha=0.7, bins=100, label='Abnormal')
    plt.legend(loc='upper right')
    plt.show()

    print(sum(x_data[:int(x_test_normal.shape[0])])/int(x_test_normal.shape[0]))
    print(sum(x_data[int(x_test_normal.shape[0]):])/int(x_test_abnormal.shape[0]))


    
    for i in range(x_test_normal.shape[0]):
        print(normalfile_list[i], sumPred[i])

    for i in range(x_test_abnormal.shape[0]):
        print(tmplist[i], sumPred[i+x_test_normal.shape[0]])
    