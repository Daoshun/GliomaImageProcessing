from keras.applications.vgg16 import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt


conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(192, 192, 3))
set_trainable = False
# print(type(conv_base))
# print(conv_base)
# include_top :是否包括顶层的全链接层 input_shape: 必须大于（48,48,3）

base_dir = '/home/vincent/data/GliomaImageProcessing'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


def extarct_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 6, 6, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(192, 192),
        batch_size=batch_size,
        class_mode='binary')

    i = 0
    for inputs_batch, labels_batch in generator:
        # print(type(inputs_batch))
        features_batch = conv_base.predict(inputs_batch)
        # print(type(features_batch))
        # print(labels_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break

    return features, labels


train_features, train_labels = extarct_features(train_dir, 80)
validation_features, validation_labels = extarct_features(validation_dir, 25)
# test_features, test_labels = extarct_features(test_dir, 1000)

train_features = np.reshape(train_features, (80, 6 * 6 * 512))
validation_features = np.reshape(validation_features, (25, 6 * 6 * 512))
# test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu', input_dim=6 * 6 * 512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

# model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
# history = model.fit(train_features, train_labels,
#                     epochs=50,
#                     batch_size=10,
#                     validation_data=(validation_features, validation_labels))
#
#
# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# plt.plot(epochs, acc, 'g', label='Training acc')
# plt.plot(epochs, val_acc, 'bo', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, loss, 'g', label='Training loss')
# plt.plot(epochs, val_loss, 'bo', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()