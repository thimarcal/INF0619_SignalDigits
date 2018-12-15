# coding: utf-8
import os
import numpy as np
from random import sample, seed
from shutil import copyfile

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,15) # Make the figures a bit bigger

from keras.preprocessing.image import load_img, img_to_array
from keras.utils import np_utils

from shutil import copyfile
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score


seed(42)
np.random.seed(42)

def splitData(datasetDir='./Dataset', nbClasses=10):
    # Initially Split Data between Train / Validation and Test
    # This data will be used during all process, and epochs will shuffle only train data
    train_files = {}
    val_files = {}
    test_files = {}
    for i in range(nbClasses):
        filenames = os.listdir(os.path.join(datasetDir, str(i)))
        shuffledFiles = sample(filenames, len(filenames))

        # 60% for Train
        # 15% for Validation
        # 25% for Test
        for index, file in enumerate(shuffledFiles):
            file = os.path.join(datasetDir, str(i), file)
            if index < len(filenames)*0.6:
                train_files[file] = i
            elif index < len(filenames)*0.75:
                val_files[file] = i
            else:
                test_files[file] = i
    return train_files, val_files, test_files

def createDatasetSplitted(files, dest_path, data_type):
    dest_path_folder = dest_path+"\\"+data_type  
    os.makedirs(dest_path_folder, exist_ok=True)
    print(dest_path_folder)
    
    for i in range(10):
        os.makedirs(dest_path_folder+"\\"+str(i), exist_ok=True)
        print(dest_path_folder+"\\"+str(i))
        
    for file in files:
        try:
            file_name = file.split("\\")
            dest_path2 = dest_path_folder+"\\"+file_name[-2]+"\\"+file_name[-1] 
            print(file, dest_path2)
            copyfile(file, dest_path2)
        except IOError as e:
            raise
	
#plot the images from imgList
def plotImages(imgListDict):
    for imgPath in imgListDict.keys():
        plotImage(imgPath)

def plotImage(imagePath, input_shape=(100,100,3)):
    print(imagePath)
    img1 = img_to_array(load_img(imagePath, target_size=input_shape))
    img1 = img1.astype('float32')
    img1 /= 255.0
    
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)

    ax.imshow(np.uint8(img1*255.0), interpolation='nearest')
    plt.show()

def getDatasetSize(dataset):
    return len(dataset)

def getLabelFromImgName(imgPath, dataset):
    return dataset.get(imgPath)
    
#Read our dataset in batches
def loadDatasetInBatches(dataset, batch_size=32, input_shape=(100,100,3), nbClasses=10):
    fileNames = dataset
        
    while True:
        imagePaths = sample(fileNames.keys(), len(fileNames)) #shuffle images in each epoch
        
        batch, labelList = [], []
        nInBatch = 0
        
        #loop of one epoch
        for idx in list(range(len(imagePaths))):
                        img = img_to_array(load_img(imagePaths[idx], target_size=input_shape))
                        img = img.astype('float32')
                        img /= 255.0
                    
                        label = np_utils.to_categorical(getLabelFromImgName(imagePaths[idx], dataset), nbClasses)
                        
                        ######### If you want to run with Data Augmentation, just uncomment here
                        ##### you can add more transformations (see https://keras.io/preprocessing/image/)
                        ### We apply a random transformation and add this image (instead of the original)
                        ### to the batch...
                        
                        #dataAugmentator = ImageDataGenerator(horizontal_flip = True)
                        #img = dataAugmentator.random_transform(img)
                        
                        
                        batch.append(img)
                        labelList.append(label)
                        nInBatch += 1
                        
                        #if we already have one batch, yields it
                        if nInBatch >= batch_size:
                            yield np.array(batch), np.array(labelList)
                            batch, labelList = [], []
                            nInBatch = 0

        #yield the remaining of the batch
        if nInBatch > 0:
            yield np.array(batch), np.array(labelList)

    return batch, labelList

from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, rotation=45)
    plt.yticks(tick_marks)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
	
# Compute confusion matrix
def plot_confusion(yTest, yTestPred, name):
    cm = confusion_matrix(yTest, yTestPred)
    np.set_printoptions(precision=2)

    # Normalize the confusion matrix by row (i.e by the number of samples in each class)
    cm_normalized = (cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])*100
    print('Classification report')
    print(classification_report(yTest, yTestPred))
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure(figsize=(5, 5))
    plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix (%s)' % (name))

    plt.show()
	
