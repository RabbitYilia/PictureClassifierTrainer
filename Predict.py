import pandas as pd
import numpy as np
import os
import tensorflow as tf
import shutil
from tensorflow import keras

# 数据文件夹
data_path = './data'
predict_path = './predict'
model_path = os.path.join("./model", "model.hdf5")
log_dir="./log"

# dimensions of our images.
img_width, img_height = 224, 224
batch_size = 1

if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = keras.models.load_model(model_path)

# this is the augmentation configuration we will use for training
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
validation_generator = test_datagen.flow_from_directory(
    predict_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator.reset()
pred = model.predict_generator(validation_generator, verbose=1)

predicted_class_indices = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)
label = dict((v,k) for k,v in labels.items())

# 建立代码标签与真实标签的关系
predictions = [label[i] for i in predicted_class_indices]

#建立预测结果和文件名之间的关系
filenames = validation_generator.filenames
for idx in range(len(filenames)):
    src = os.path.join(predict_path,filenames[idx])
    dst = src.replace(predict_path,"./result").replace("Raw\\",predictions[idx]+"\\")
    shutil.move(src, dst)


