# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:38:24 2016

@author: Aditya
"""

#***********************NON-CAUSAL*****************************

#***account for missing data by using original image splices***

#***LIBRARIES***
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#***TRAINING FUNCTION***
def trainRF(img, w, h, g, initChoice):
    im = Image.open(img) #load image
    im = im.convert('1') #convert image to black and white
    imgRes = np.array(im.size) #image resolution (rows, columns)
    Npix = list(im.getdata()) #load pixels as a list
    dict = Counter(Npix) #volume fraction
    print ("original image black pixels:", (dict[0]))
    print ("original image white pixels:", (dict[255]))
    pixChk = im.load() #load pixels for validation
    
    #***VALIDATION BY OBSERVATION***
    print(imgRes) # (rows, columns)
    #print(Npix[158])
    #print(pixChk[158,0])
    
    #***CREATE PIXEL MATRIX BY SPLITTING THE Npix LIST***
    pixMat = np.zeros(shape = imgRes+2*g)
    num = 0
    for i in range(0,imgRes[1]):
        pixMat[g:imgRes[0]+g,i+3] = Npix[num:num+imgRes[0]]
        num = num + imgRes[0]
        
    #***ADD PIXEL SPLICES TO ACCOUNT FOR MISSING DATA***
    for i in range(0,g):
        pixMat[i,:] = pixMat[i+g,:]
        pixMat[:,i] = pixMat[:,i+g]
        pixMat[imgRes[0]+2*g-1-i,:] = pixMat[imgRes[0]+g-1-i,:]
        pixMat[:,imgRes[1]+2*g-1-i] = pixMat[:,imgRes[1]+g-1-i]
        
    pixMat = pixMat/255
    
    #***CHECK***    
    #print(pixMat[:,156])


    #***FORM X AND Y ARRAYS ON THE ORIGINAL IMAGE***
    X, Y = generateXY(pixMat, imgRes, g, w, h)
            
    #***USE RANDOM FOREST CLASSIFIER***
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(Y,X)
    
    if(initChoice==1):
        #***CREATE BLACK INITIAL IMAGE***
        print("Initial pixels black")
        initImg = np.zeros(shape = imgRes+2*g)        
        IX, IY = generateXY(initImg, imgRes, g, w, h)
        
    elif(initChoice==2):
        #***CREATE WHITE INITIAL IMAGE***
        print("Initial pixels white")
        initImg = np.ones(shape = imgRes+2*g)        
        IX, IY = generateXY(initImg, imgRes, g, w, h)
        
    elif(initChoice==3):
        #***CREATE RANDOM INITIAL IMAGE***
        print("Initial pixels random")
        initImg = np.random.randint(2, size = imgRes+2*g)        
        IX, IY = generateXY(initImg, imgRes, g, w, h)
        
    else:
        #***CREATE INITIAL IMAGE WITH SLICES***
        initImg = np.zeros(shape = imgRes+2*g)
        print("Initial pixels splices of original image")
        initImg[0,:] = pixMat[0,:]
        end = int(np.ceil((imgRes[0]+2*g-1)/2))
        for i in range(1,end+1):
            if i==end:
                if np.mod(imgRes[0]+2*g-1,2) == 0:
                    initImg[2*i-1,:] = pixMat[1,:]
                    initImg[2*i,:] = pixMat[0,:]
                else:
                    initImg[2*i-1,:] = pixMat[1,:]
            else:
                initImg[2*i-1,:] = pixMat[1,:]
                initImg[2*i,:] = pixMat[0,:]
        IX, IY = generateXY(initImg, imgRes, g, w, h)
            
    
    initImg = np.zeros(shape = imgRes+2*g)        
    IX, IY = generateXY(initImg, imgRes, g, w, h)
    
    #***RECONSTRUCT THE STATISTICAL EQUIVALENT IMAGE***
    temp = 1    
    while temp==1:
        count = 0
        cw = 0
        cb = 0
        for i in range(g,imgRes[0]+g):
            for j in range(g,imgRes[1]+g):
                Wprob = clf.predict_proba(IY[count].reshape(1,-1))
                a = np.random.uniform(0,1,1)
                if a[0] < Wprob[0][0]:
                    initImg[i,j] = 1 #white
                    cw = cw + 1
                else:
                    initImg[i,j] = 0 #black
                    cb = cb + 1
                IX, IY = generateXY(initImg, imgRes, g, w, h)
                #print(count)              
                #print(count)                
                count = count + 1
        print("reconstructed white pixels:",cw)
        print("reconstructed black pixels:",cb)
        #n_samples = imgRes[0]*imgRes[1]        
        #tree = KDTree(n_samples, 2)
        #dist, ind = tree.query(n_samples[0], k=3)
        #print(dist)
        #G = np.zeros((imgRes[0],imgRes[1],3))
        #G[initImg==1] = [1,1,1]
        #G[initImg==0] = [0,0,0]
        
        #plt.imshow(G,interpolation='nearest')
        #plt.show()
        temp = 2
    
#***FUNCTION THAT RETURNS 1 FOR WHITE AND 0 FOR BLACK
#def blackWhite(pixMat,i,j):
#    return pixMat[i,j]/255
    
def generateXY(imgMat, imgRes, g, w, h):
    #***SUPERVISED DATA***
    X = [] #input
    Y = [] #output
    neighPix = []
    
    for i in range(g,imgRes[0]+g):
        for j in range(g,imgRes[1]+g):
            X.append(imgMat[i,j])
            #neighborhood pixels
            for k in range(i-w,i+w+1):#non-causal neighbors
                for l in range(j-h,j+h+1):
                    if k!=i or l!=j:
                        neighPix.append(imgMat[k,l])
            Y.append(neighPix)
            neighPix = []
    return np.array(X), np.array(Y)
            
            
#trainRF(image_name, w, h, extent of green region on each side(g))
#g should always be greater than w and h
#initChoice = 1 (INITIAL PIXELS BLACK)
#initChoice = 1 (INITIAL PIXELS WHITE)
#initChoice = 1 (INITIAL PIXELS RANDOM)
#initChoice = 1 (INITIAL PIXELS SLICES OF ORIGINAL IMAGE)
clf = trainRF("img2.png",2,2,3,4)