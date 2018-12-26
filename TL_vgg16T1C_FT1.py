from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt


train_dir = r'/home/vincent/data/GliomaImageProcessing/train'
validation_dir = r'/home/vincent/data/GliomaImageProcessing/validation'
test_dir = r'/home/vincent/data/GliomaImageProcessing/test'

model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(192, 192, 3))  # 192,192,3
set_trainable = False
# for layer in model_vgg16.layers:
#     layers.trainable = False
for layer in model_vgg16.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
        # print(layer.name)
    if set_trainable:
        layer.trainable = True
        # print(layer.name)
model_vgg16.summary()
# print(type(model_vgg16))

# x = model_vgg16.output
# # print(type(x))
# # print(x)
# x = Flatten()(x)
# x = Dense(256, activation='relu')(x)
# x = Dense(256, activation='relu')(x)
# predictions = Dense(1, activation='sigmoid')(x)
#
# # this is the model we will train
# model = Model(inputs=model_vgg16.input, outputs=predictions)
# model.summary()


model = models.Sequential()
model.add(model_vgg16)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

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
    batch_size=5,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    directory=validation_dir,
    target_size=(192, 192),
    batch_size=5,
    class_mode='binary')

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-5),
#               metrics=['acc'])

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-6),
              metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=5,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=5)


# model.save('cats_and_dogs_small_5.h5')
# model.load_weights('cats_and_dogs_small_4.h5')


# def smooth_curve(points, factor=0.8):
#     smoothed_points = []
#     for point in points:
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(int(previous * factor + points * (1 - factor)))
#         else:
#             smoothed_points.append(point)


acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'go', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# epochs = range(1, len(acc) + 1)
# plt.plot(epochs, smooth_curve(acc), 'bo', label='Smoothed training acc')
# plt.plot(epochs, smooth_curve(val_acc), 'b', label='Smoothed validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
#
# plt.figure()
#
# plt.plot(epochs, smooth_curve(loss), 'bo', label='Smoothed training loss')
# plt.plot(epochs, smooth_curve(val_loss), 'b', label='Smoothed validation loss')
# plt.title('Training and validation loss')
# plt.legend()
#
# plt.show()

# test_generator = test_datagen.flow_from_directory(
#     test_dir,
#     target_size=(150, 150),
#     batch_size=20,
#     class_mode='binary'
# )
# test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
# print('test acc:', test_acc)