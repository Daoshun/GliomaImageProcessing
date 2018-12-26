from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
import matplotlib.pyplot as plt


train_dir = r'/home/vincent/data/GliomaImageProcessing/train'
validation_dir = r'/home/vincent/data/GliomaImageProcessing/validation'
test_dir = r'/home/vincent/data/GliomaImageProcessing/test'

model_inception = InceptionV3(include_top=False, weights='imagenet', input_shape=(192, 192, 3))
# model_vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(192, 192, 3))  # 192,192,3
set_trainable = False
def setup_to_fine_tune(model, base_model):
    GAP_LAYER = 17
    # max_pooling_2d_2
    for layer in base_model.layers[:GAP_LAYER+1]:
        layer.trainable = False
    for layer in base_model.layers[GAP_LAYER+1:]:
        layer.trainable = True
    model.compile(optimizer=optimizers.Adagrad(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
# # for layer in model_vgg16.layers:
# #     layers.trainable = False
# for layer in model_vgg16.layers:
#     if layer.name == 'block5_conv1':
#         set_trainable = True
#         # print(layer.name)
#     if set_trainable:
#         layer.trainable = True
#         # print(layer.name)
model_inception.summary()
# print(type(model_vgg16))


model = models.Sequential()
model.add(model_inception)
model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l1(0.001)))
# model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), activity_regularizer=regularizers.l1(0.001)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()
setup_to_fine_tune(model, model_inception)

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

# model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.RMSprop(lr=1e-6),
#               metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=5,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=5)

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
