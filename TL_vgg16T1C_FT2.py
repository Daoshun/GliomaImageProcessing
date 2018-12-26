import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import matplotlib.pyplot as plt
# dimensions of our images.
img_width, img_height = 192, 192

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = r'/home/vincent/data/GliomaImageProcessing/train'
validation_data_dir = r'/home/vincent/data/GliomaImageProcessing/validation'

nb_train_samples = 80
nb_validation_samples = 25
epochs = 30
batch_size = 10


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load('bottleneck_features_validation.npy')
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # history = model.fit(train_data, train_labels,
    #                     epochs=epochs,
    #                     batch_size=batch_size,
    #                     validation_data=(validation_data, validation_labels))

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)

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

save_bottlebeck_features()
train_top_model()

# model.summary()
#
# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest')
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_directory(
#     directory=train_dir,
#     target_size=(192, 192),
#     batch_size=20,
#     class_mode='binary')
#
# validation_generator = test_datagen.flow_from_directory(
#     directory=validation_dir,
#     target_size=(192, 192),
#     batch_size=20,
#     class_mode='binary')

