import os
import tensorflow as tf
from tensorflow import keras


# 数据文件夹
data_path = './data'
model_path = os.path.join("./model", "model.hdf5")
log_dir="./log"
n_categories = 10
batch_size=  1
img_width, img_height, c = 224, 224, 3
epochs = 100
nb_train_samples = 5000
nb_validation_samples = nb_train_samples / 5

if(os.path.exists(model_path)==False):
    base_model = keras.applications.VGG16(
        weights='imagenet',
        include_top=False,
        input_tensor=keras.layers.Input(shape=(img_width, img_height, c)))
    for layer in base_model.layers:
        layer.trainable = False
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(1024, activation='relu')(x)
    prediction = keras.layers.Dense(n_categories, activation='softmax')(x)
    model = keras.models.Model(inputs=base_model.input, outputs=prediction)
    model.compile(
        optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    model.save(model_path)

model = keras.models.load_model(model_path)
checkpoint = keras.callbacks.ModelCheckpoint(
    model_path,
    monitor='val_acc',
    verbose=1,
    save_best_only=True,
    mode='max',
    save_weights_only=False,
)
callbacks_list = [checkpoint, keras.callbacks.CSVLogger("Log"+'.csv')]
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    data_path,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
model.summary()
hist = model.fit_generator(
    generator=train_generator,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    callbacks=callbacks_list)
