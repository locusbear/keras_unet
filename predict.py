import cv2
import PIL

from tensorflow.keras import layers
import numpy as np 

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model

def predict():
	img_test_npy=''#导入的test img npy文件路径
	label_test_npy=''#导入的test label npy文件路径

	img_test=np.load(img_test_npy)
	label_test=np.load(label_test_npy)
	print(img_test.shape)
	print(label_test.shape)

	model=load_model('')#模型权重路径
	pre_img=model.predict(x=img_test,batch_size=1,verbose=3)
	print(pre_img.shape)

	pre_img_argmax=np.argmax(pre_img,axis=3)

	print(pre_img_argmax.shape)

	for i in range(0,20,2):
		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(img_test[i])
		plt.subplot(2,2,2)
		plt.imshow(label_test[i])
		
		plt.subplot(2,2,3)
		plt.imshow(img_test[i+1])
		
		plt.subplot(2,2,4)
		plt.imshow(pre_img_argmax[i+1])
		plt.show()

predict()





