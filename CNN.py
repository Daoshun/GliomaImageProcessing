from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

train_dir = r'/home/vincent/data/GliomaImageProcessing/train'
validation_dir = r'/home/vincent/data/GliomaImageProcessing/validation'
test_dir = r'/home/vincent/data/GliomaImageProcessing/test'

# 调整像素值
# train_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(208, 208), batch_size=5,
                                                    class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(directory=validation_dir, target_size=(208, 208),
                                                              batch_size=5, class_mode='binary')
# 32,64,128,128
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(208, 208, 3)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu',
#                        kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.001)))
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
# 二类问题多用sigmoid函数，-1到1,多类问题用softmax强化概率大的值
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])
# model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['acc'])
history = model.fit_generator(train_generator, steps_per_epoch=5, epochs=30, validation_data=validation_generator,
                              validation_steps=5)

# model.save('cnn.h5')

#载入模型
# model = load_model('cats_and_dogs_small_1.h5')
# result = model.evaluate(test_data)
# print('\nTest Acc', result[1])

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
