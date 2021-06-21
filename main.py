import os
import random
from idlelib import history
import matplotlib.pyplot as plt  # plt 用于显示图片
import PIL
from tensorflow.python.keras.preprocessing import image
import matplotlib.image as mpimg  # mpimg 用于读取图片
from PIL import Image

import numpy as np
from keras.layers import Dense
from keras.models import Model
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.preprocessing.image import load_img
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report

# from keras.utils import np_utils
# from keras.utils.np_utils import *
from keras.utils.np_utils import to_categorical  # 做one_hot编码
from keras.applications.inception_resnet_v2 import InceptionResNetV2

class_names_to_ids = {'cardboard': 0, 'glass': 1, 'metal': 2, 'paper': 3, 'plastic': 4, 'trash': 5}

data_dir = 'dataset-resized/'
output_path = 'list.txt'
fd = open(output_path, 'w')
for class_name in class_names_to_ids.keys():
    images_list = os.listdir(data_dir + class_name)
    for image_name in images_list:
        fd.write('{}/{} {}\n'.format(class_name, image_name, class_names_to_ids[class_name]))
fd.close

# _NUM_VALIDATION = 505
_NUM_VALIDATION = 50
_RANDOM_SEED = 0
list_path = 'list.txt'
train_list_path = 'list_train.txt'
val_list_path = 'list_val.txt'
fd = open(list_path)
lines = fd.readlines()
fd.close()
random.seed(_RANDOM_SEED)
random.shuffle(lines)
fd = open(train_list_path, 'w')
for line in lines[_NUM_VALIDATION:100]:
    fd.write(line)
fd.close()
fd = open(val_list_path, 'w')
for line in lines[:_NUM_VALIDATION]:
    fd.write(line)
fd.close()


def get_train_test_data(list_file):
    list_train = open(list_file)
    x_train = []  # 获取的路径
    y_train = []  # 图像的标签
    for line in list_train.readlines():
        x_train.append(line.strip()[:-2])
        y_train.append(int(line.strip()[-1]))
        # print(line.strip())
    return x_train, y_train


x_train, y_train = get_train_test_data('list_train.txt')
x_test, y_test = get_train_test_data('list_val.txt')


def process_train_test_data(x_path):
    images = []
    for image_path in x_path:
        img_load = load_img('dataset-resized/' + image_path)
        img = image.img_to_array(img_load)
        img = preprocess_input(img)
        images.append(img)
    return images


train_images = process_train_test_data(x_train)
test_images = process_train_test_data(x_test)

base_model = InceptionResNetV2(include_top=False, pooling='avg')
outputs = Dense(6, activation='softmax')(base_model.output)
model = Model(base_model.inputs, outputs)
# 设置ModelCheckpoint，按照验证集的准确率进行保存准确率比较大的模型from sklearn.metrics import classification_report
save_dir = 'train_model'
filepath = "model_{epoch:02d}-{accuracy:.2f}.hdf5"  # 保留测试时第几代，验证准确率（保留两位小数）
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='accuracy', verbose=1,
                             save_best_only=True)  # 监控测试集准确率，将结果输出


# 模型设置
def acc_top3(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)


def acc_top5(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy', acc_top3, acc_top5])  # 使用adam优化器，categorical_crossentropy交叉熵损失，监控top3和top5的准确率
# 模型训练

'''
model.fit(np.array(train_images), to_categorical(y_train), batch_size=8, epochs=5, shuffle=True,
          validation_data=(np.array(test_images), to_categorical(y_test)), callbacks=[checkpoint])
'''
model.load_weights('train_model/model_01-0.33.hdf5')  # 添加路径加载图片进行预测

y_pred = model.predict(np.array(test_images))

# pic.shape  # (512, 512, 3)
labels={0:'cardboard',1:'glass',2:'metal',3:'paper',4:'plastic',5:'trash'}

plt.figure(figsize=(20, 16))
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.title('pred:%s/truth:%s' % (labels[np.argmax(y_pred[i])], labels[y_test[i]]))
    pic = mpimg.imread('dataset-resized/' + x_test[i])
    plt.imshow(pic)

'''
for item in zip(y_test,ans):
    print(item)
    
    
'''


"""
可以使用 Python 中一个名为 imgaug 的图像增广库，即对原来的数据集
进行旋转，缩放，对称变换，亮度变换，高斯模糊等一系列操作，
在扩充数据集规模的同时，也能让算法模型对不同的场景均有较好的鲁棒性。

"""