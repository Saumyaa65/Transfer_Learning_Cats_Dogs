import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_dir="training_set"
test_dir="test_set"
img_shape=(128,128,3)

base_model=tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                    include_top=False, weights='imagenet')
# img shape as the pretrained model expects the input in particular shape
# include_top False as we dont want last layer of pretrained model as our
# model output is different. weights imagenet as network trained on imagenet.

# freeze model as we dont want to change previous layers and do custom
# training on only the layers that we add later.
base_model.trainable=False

# we need to define custom head for network to mkae it suitable for our task.
# current base_model.output= (None, 4, 4, 1280) we need to change shape.
global_average_layer=tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
# this layer is similiar to maxpool layer
# global_average_layer.shape= (None, 1280)
prediction_layer=tf.keras.layers.Dense(units=1,
                                    activation='sigmoid')(global_average_layer)
model = tf.keras.models.Model(inputs=base_model.input, outputs=prediction_layer)
print(model)
# this line creates a combined network from the 2 models
opt=tf.keras.optimizers.RMSprop(learning_rate=0.0001)
# low learning rate as we are using pre-trained model
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
print(model)

data_gen_train=ImageDataGenerator(rescale=1/255.0)
data_gen_test=ImageDataGenerator(rescale=1/255.0)
train_generator=data_gen_train.flow_from_directory(directory=training_dir,
                target_size=(128,128), batch_size=128, class_mode='binary')
test_generator=data_gen_test.flow_from_directory(directory=test_dir,
                target_size=(128,128), batch_size=128, class_mode='binary')
# model.fit(train_generator, epochs=5,validation_data=test_generator)

base_model.trainable=True
# len(base_model.layers)=155
fine_tune_at=100
# we'll train layers after 100 and freeze the previous layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable=False
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=5,validation_data=test_generator)
