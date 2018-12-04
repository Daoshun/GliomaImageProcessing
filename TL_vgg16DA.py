from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt

train_dir = r'/home/vincent/data/GliomaImageProcessing/train'
validation_dir = r'/home/vincent/data/GliomaImageProcessing/validation'
test_dir = r'/home/vincent/data/GliomaImageProcessing/test'

model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(192, 192, 3))
set_trainable = False
model = models.Sequential()
model.add(model_vgg16)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()


# 调整像素值
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(192, 192),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(192, 192),
    batch_size=20,
    class_mode='binary')

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=10)

# model.save('TL_vgg16DA.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary'
# )
# test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
# print('test acc:', test_acc)