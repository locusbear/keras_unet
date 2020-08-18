import cv2
import os
import random
import glob
import numpy as np 
from keras.preprocessing import image
#author cc 2020-8-10

'''
1 使用cv2预处理函数处理img
2 合并img和label并重命名1.2.3...
3 使用keras函数对合并图像进行数据增强
4 分开img和label
5 保存数据集为npy形式
'''

'''
初版工作已经完成，后续需要优化代码，只需要一个主函数，通过修改参数来确定那些函数要运行，那些函数不运行。
'''








def merge(img_path,img_label_path,img_name):
	label_path=''#原始数据的label路径

	path_exist(img_label_path)

	for img_idx in img_name:
		img=cv2.imread(img_path+img_idx)
		print(img.shape)
		label=cv2.imread(label_path+img_idx)
		#print(img.shape)
		#print(label.shape)
		img_label=img
		img_label[:,:,0]=label[:,:,0]

		cv2.imwrite(img_label_path+img_idx,img_label)

def k_DataAugmentation(img_label_path,img_name,data_aug_path):
	


	path_exist(data_aug_path)


	datagen=image.ImageDataGenerator(featurewise_center=True,
		featurewise_std_normalization=True,
		samplewise_center=True,
		samplewise_std_normalization=True,
		rotation_range=5,
		width_shift_range=0.2,
		height_shift_range=0.2,
		rescale=1./255,
		shear_range=0.2,
		zoom_range=0.5,
		horizontal_flip=True,
		fill_mode='nearest')
	for img_inx in img_name:
		i=0
		path_exist(data_aug_path+img_inx.split('.png')[0])

		img=cv2.imread(img_label_path+img_inx)
		#print(img.shape)
		img=img.reshape((1,)+img.shape)
		for batch in datagen.flow(img,batch_size=1,save_to_dir=data_aug_path+img_inx.split('.png')[0],save_prefix=img_inx.split('.png')[0],save_format='png'):
			i+=1
			if i>71:
				break

def train_label(data_aug_path,img_name):
	img_aug_path=''#数据扩充并分离融合图像后的img
	label_aug_path=''#数据扩充并分离融合图像后的label
	path_exist(img_aug_path)
	path_exist(label_aug_path)
	for path_inx in img_name:
		#for img_inx in (data_aug_path+path_inx.split('.png')[0]):
		#img_floder=os.listdir(data_aug_path+path_inx.split('.png')[0])
		path_exist(img_aug_path+path_inx.split('.png')[0])
		path_exist(label_aug_path+path_inx.split('.png')[0])

		img_floder=glob.glob(data_aug_path+path_inx.split('.png')[0]+'/*.'+'png')


		for img_inx in img_floder:
			img=cv2.imread(img_inx)

			train_img=img[:,:,1]
			label_img=img[:,:,2]#本来label是存储在0通道的，但是中途某个地方做了通道数的改变，应该是向后移动了一位。从原始Keras代码中也发现了这一点。
			label_img[label_img>125]=255
			label_img[label_img<=125]=0


			cv2.imwrite(img_aug_path+path_inx.split('.png')[0]+'/'+img_inx.split('\\')[-1],train_img)
			cv2.imwrite(label_aug_path+path_inx.split('.png')[0]+'/'+img_inx.split('\\')[-1],label_img)
			
def generate_npy():
	#1 打乱数据集 2 划分训练集验证集和测试集 3保存为npy文件 4验证保存npy文件的正确性
	img_aug_path=''#同上
	label_aug_path=''#同上

	img_floder=os.listdir(img_aug_path)
	img_list=[]
	label_list=[]
	for path_inx in img_floder:
		img_inx=glob.glob(img_aug_path+path_inx+'/*.'+'png')
		for inx in img_inx:
			img=cv2.imread(inx)
			label=cv2.imread(inx.replace('aug_train','aug_label'))#需要验证os.replace的用法是否正确。(是正确的)
			img_list.append(img)
			label_list.append(label)

	assert len(img_list)==len(label_list)
	#print(len(img_list))#8608
	img_list=np.array(img_list)
	label_list=np.array(label_list)

	index=[i for i in range(len(img_list))]
	random.shuffle(index)
	img=img_list[index]
	label=label_list[index]

	#print("img",img.shape)#(8608, 256, 256, 3)
	#print('label',label.shape)#(8608, 256, 256, 3)
	#因为keras可以读取npy文件，所以转换成npy文件
	img_npy=''#保存后的img npy格式文件
	label_npy=''#保存后的label npy格式文件


	np.save(img_npy,img)
	np.save(label_npy,label)

def test_npy():
	test_npy=''#测试的img npy路径
	test_label_npy=''
	test_img=''#原始测试img
	test_label=''#原始测试label

	test_img_path=glob.glob(test_img+'/*.'+'png')
	test_label_path=glob.glob(test_label+'/*.'+'png')

	test_list=[]
	test_label_list=[]

	for img_inx in test_img_path:
		img=cv2.imread(img_inx)
		label=cv2.imread(img_inx.replace("test","test_label"))
		test_list.append(img)
		test_label_list.append(label)
	assert len(test_list)==len(test_label_list)

	test_list=np.array(test_list)
	test_label_list=np.array(test_label_list)

	np.save(test_npy,test_list)
	np.save(test_label_npy,test_label_list)




def path_exist(path):#如果该路径下不存在该文件夹，则创建文件夹。
	if not os.path.exists(path):
		os.makedirs(path)

if __name__=='__main__':
	img_label_path=''#获取img与label融合后的图像
	img_path=''#原始训练img路径
	img_name=os.listdir(img_path)

	data_aug_path=''#数据扩充后的融合图像



	merge(img_path,img_label_path,img_name)#把img和label融合成一张图像
	k_DataAugmentation(img_label_path,img_name)#对融合后的图像做数据增强
	train_label(data_aug_path,img_name)#把增强后的图像重新划分train和label	
	generate_npy()#生成训练npy
	test_npy()#导入预测npy
