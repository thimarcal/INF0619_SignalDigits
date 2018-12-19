# coding: utf-8
import os
import numpy as np
from random import sample, seed

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15,15) # Make the figures a bit bigger

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import np_utils


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
def loadDatasetInBatches(dataset, batch_size=32, input_shape=(100,100,3), nbClasses=10, shouldAugmentData=False):
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
                        if shouldAugmentData:
                            dataAugmentator = ImageDataGenerator(horizontal_flip = True, rotation_range=20)
                            img = dataAugmentator.random_transform(img)
                        
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
