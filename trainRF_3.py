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
    print("\n\nEXTRACTING PIXELS...\n\n")
    im = Image.open(img) #load image
    im = im.convert('1') #convert image to black and white
    plt.imshow(im, cmap='Greys',  interpolation='nearest')
    plt.show()
    print("Original Image\n")
    imgRes = np.array(im.size) #image resolution (rows, columns)
    Npix = list(im.getdata()) #load pixels as a list
    dict = Counter(Npix) #volume fraction
    print ("original image black pixels:", (dict[0]))
    print ("original image white pixels:\n", (dict[255]))
    pixChk = im.load() #load pixels for validation
    
    #***VALIDATION BY OBSERVATION***
    print(imgRes) # (rows, columns)
    #print(Npix[32])
    #print(pixChk[32,0])
    
    #***CREATE PIXEL MATRIX BY SPLITTING THE Npix LIST***
    print("\n\nCREATING PIXEL MATRIX...\n\n")
    pixMat = np.zeros(shape = imgRes+2*g)
    num = 0
    for i in range(0,imgRes[1]):
        pixMat[g:imgRes[0]+g,i+g] = Npix[num:num+imgRes[0]]
        num = num + imgRes[0]
        
    #***ADD PIXEL SPLICES TO ACCOUNT FOR MISSING DATA***
    print("\n\nADDING PIXEL SPLICES TO ACCOUNT MISSING DATA...\n\n")
    for i in range(0,g):
        pixMat[i,:] = pixMat[i+g,:]
        pixMat[:,i] = pixMat[:,i+g]
        pixMat[imgRes[0]+2*g-1-i,:] = pixMat[imgRes[0]+g-1-i,:]
        pixMat[:,imgRes[1]+2*g-1-i] = pixMat[:,imgRes[1]+g-1-i]
        
    pixMat = pixMat/255
    
    #***CHECK***    
    #print(pixMat[:,156])


    #***FORM X AND Y ARRAYS ON THE ORIGINAL IMAGE***
    print("\n\nTRAINING THE RANDOM FOREST...\n\n")
    X, Y = generateXY(pixMat, imgRes, g, w, h)
            
    #***USE RANDOM FOREST CLASSIFIER***
    clf = RandomForestClassifier(n_estimators=10)
    clf = clf.fit(Y,X)
    
    print("\n\nCREATING THE INITIAL IMAGE...\n\n")
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
            
    plt.imshow(initImg, cmap='Greys',  interpolation='nearest')
    plt.show()
    print("Initial Image")
    
    #***RECONSTRUCT THE STATISTICAL EQUIVALENT IMAGE***  
    print("\n\nCREATING THE EQUIVALENT IMAGE...\n\n")
    c_adjust = 0
    cw = 0
    cb = 0
    Itr = 1
    #while abs(cw - dict[255]) > 5:        
    while Itr == 1:
        count = 0
        cw = 0
        cb = 0
        for i in range(g,imgRes[0]+g):
            for j in range(g,imgRes[1]+g):
                Wprob = clf.predict_proba(IY[count].reshape(1,-1))
                p = Wprob[0][0]
                p = p + c_adjust*np.sqrt(p*(1-p))
                a = np.random.uniform(0,1,1)
                if a[0] < p:
                    initImg[i,j] = 1 #white
                    cw = cw + 1
                else:
                    initImg[i,j] = 0 #black
                    cb = cb + 1
                #print(count)              
                #print(count)                
                count = count + 1
                IX, IY = generateXY(initImg, imgRes, g, w, h)
        print("reconstructed black pixels:",cb)        
        print("reconstructed white pixels:",cw)
        print(dict[255])
        if cw > dict[255]:
            c_adjust = Itr * -0.1
        elif cw < dict[255]:
            c_adjust = Itr * 0.1
        Itr = Itr + 1
    
    #***CREATE THE PIXEL MATRIX FROM ARRAY
    print("\n\nPLOTTING THE NEW IMAGE...\n\n")
    new_im = np.zeros(imgRes)
    new_im[0:imgRes[0],0:imgRes[1]] = initImg[g:imgRes[0]+g, g:imgRes[1]+g]
    plt.imshow(new_im, cmap='Greys',  interpolation='nearest')
    plt.show()
    print("Reconstructed Image")
        

    
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

#***FUNCTION CALL***
trainRF("img4.png",2,2,3,4)
