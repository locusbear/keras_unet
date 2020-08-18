'''
1 导入npy数据并验证train和label是否能对应上
2 搭建Unet模型
3 写入超参数，损失函数等等
'''
import cv2
import PIL
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np 
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
np.set_printoptions(threshold=np.inf)

def load_npy():
	img_npy=''#导入保存成npy格式的img
	label_npy=''#导入保存成npy格式的label

	img_train=np.load(img_npy)
	img_label=np.load(label_npy)


	#for i in range(img_label):



	#img_train/=255
	#mean=img_train.mean(axis=0)
	#img_train-=mean#这里的预处理方式不清楚是否恰当--前面数据扩充的的时候进行了类似的操作但是没有效果
	#img_label=img_label/255
	print('load npy is finsh')


	


	#验证train和label是否能一一对应
	# 221 222 223 224代表四幅图像的位置，以此类推。


	'''
	for i in range(0,20,2):
		plt.figure()
		plt.subplot(2,2,1)
		plt.imshow(img_train[i])
		plt.subplot(2,2,2)
		plt.imshow(img_label[i])
		
		plt.subplot(2,2,3)
		plt.imshow(img_train[i+1])
		
		plt.subplot(2,2,4)
		plt.imshow(img_label[i+1])
		plt.show()
	'''
	return img_train,img_label

class  bulid_network():

	def __init__(self,img_train,img_label):
		self.train=img_train
		self.label=img_label
		print(self.label.shape)


	def unet(self):
		'''
		使用functional api创建模型（因为Sequntial模型可能不支持skip connection）
		'''
		train_shape_1,train_shape_2,train_shape_3=self.train.shape[1],self.train.shape[2],self.train.shape[3]
		train_inputs=keras.Input(shape=(train_shape_1,train_shape_2,train_shape_3),name='img')
		'''
		ValueError: Input 0 of layer conv2d is incompatible with the layer: expected ndim=4, found ndim=5. Full shape received: [None, 8608, 256, 256, 3]

		'''
		print('train_inputs',train_inputs.shape)

		x_conv1=layers.Conv2D(64,3,activation='relu',padding='same')(train_inputs)
		x_conv2=layers.Conv2D(64,3,activation='relu',padding='same')(x_conv1)

		x_pool1=layers.MaxPooling2D(2)(x_conv2)
		print(x_pool1.shape)

		x_conv3=layers.Conv2D(128,3,activation='relu',padding='same')(x_pool1)
		x_conv4=layers.Conv2D(128,3,activation='relu',padding='same')(x_conv3)

		x_pool2=layers.MaxPooling2D(2)(x_conv4)


		x_conv5=layers.Conv2D(256,3,activation='relu',padding='same')(x_pool2)
		x_conv6=layers.Conv2D(256,3,activation='relu',padding='same')(x_conv5)

		x_pool3=layers.MaxPooling2D(2)(x_conv6)

		x_conv7=layers.Conv2D(512,3,activation='relu',padding='same')(x_pool3)
		x_conv8=layers.Conv2D(512,3,activation='relu',padding='same')(x_conv7)

		x_pool4=layers.MaxPooling2D(2)(x_conv8)


		x_conv9=layers.Conv2D(1024,3,activation='relu',padding='same')(x_pool4)
		x_conv10=layers.Conv2D(1024,3,activation='relu',padding='same')(x_conv9)
		#print(x_conv10.shape)

		#先进行反卷积再进行跳跃连接

		x_deconv1=layers.Conv2DTranspose(512,kernel_size=(2,2),strides=(2,2))(x_conv10)
		#print(x_conv8.shape,x_deconv1.shape)
		x_conatenate1=layers.concatenate([x_conv8,x_deconv1])

		x_conv11=layers.Conv2D(512,3,activation='relu',padding='same')(x_conatenate1)
		x_conv12=layers.Conv2D(512,3,activation='relu',padding='same')(x_conv11)


		x_deconv2=layers.Conv2DTranspose(256,kernel_size=(2,2),strides=(2,2))(x_conv12)
		x_conatenate2=layers.concatenate([x_conv6,x_deconv2])

		x_conv13=layers.Conv2D(512,3,activation='relu',padding='same')(x_conatenate2)
		x_conv14=layers.Conv2D(512,3,activation='relu',padding='same')(x_conv13)


		x_deconv3=layers.Conv2DTranspose(128,kernel_size=(2,2),strides=(2,2))(x_conv14)
		x_conatenate3=layers.concatenate([x_conv4,x_deconv3])

		x_conv15=layers.Conv2D(128,3,activation='relu',padding='same')(x_conatenate3)
		x_conv16=layers.Conv2D(128,3,activation='relu',padding='same')(x_conv15)


		x_deconv4=layers.Conv2DTranspose(128,kernel_size=(2,2),strides=(2,2))(x_conv16)
		x_conatenate4=layers.concatenate([x_conv2,x_deconv4])

		x_conv17=layers.Conv2D(64,3,activation='relu',padding='same')(x_conatenate4)
		x_conv18=layers.Conv2D(64,3,activation='relu',padding='same')(x_conv17)

		outputs=layers.Conv2D(2,1,activation='softmax')(x_conv18)
		#print(outputs.shape)

		model=keras.models.Model(inputs=train_inputs,outputs=outputs)
		model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
		print('model conplie')

		return model

	def train_generator(self,batch_size):
		assert self.train.shape[0]==self.label.shape[0]
		train_length=self.train.shape[0]
		one_round=train_length//batch_size+1

		while True:
			for i in range(one_round):
				begin=i*batch_size
				end=(i+1)*batch_size
				if end>train_length:
					end=train_length

				train_inputs=self.train[begin:end]
				
				train_labels=self.label[begin:end,:,:,0:1]/255
				#print("train_labels",train_labels.shape)

				#train_labels=to_categorical(train_labels,num_classes=2)
				#print(self.label.shape)
				train_labels=to_categorical(train_labels,num_classes=2)
				train_labels=train_labels.reshape(batch_size,256,256,-1)
				yield(train_inputs,train_labels)


	def train_net(self):
		model=self.unet()
		#print(self.label.shape)
		#self.label=self.label[:,:,:,0:1]/255
		#print('33333',self.label.shape)
		#label_shape=self.label.shape
		#print(type(self.label))
		#self.label=to_categorical(self.label,num_classes=2)
		#self.label=self.label.reshape(label_shape[0],label_shape[1],label_shape[2],2)


		#model_checkpoint=ModelCheckpoint("./backup/weight:{epoch:02d}-{loss:2f}.h5",monitor='loss',verbose=2,save_best_only=True)
		model_checkpoint=ModelCheckpoint("",monitor='loss',verbose=2,save_best_only=True)#权重保存路径

		print('fit model ')
		#hist=model.fit(x=self.train,y=self.label,validation_split=0.1,batch_size=2,epoch=10,verbose=1,shuffle=True,callbacks=[model_checkpoint])
		hist=model.fit_generator(self.train_generator(batch_size=8),steps_per_epoch=1076,epochs=1,verbose=1,shuffle=True,callbacks=[model_checkpoint])

		print(hist.history)
		print ('*'*30)
		print ('Fitting model done')















img_train,img_label=load_npy()

a=bulid_network(img_train,img_label)

a.train_net()