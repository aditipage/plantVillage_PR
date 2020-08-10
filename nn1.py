import csv
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
from tensorflow.keras import datasets, layers, models, losses, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, BatchNormalization, Flatten, Dropout, Conv2D, MaxPool2D, Input
from keras.callbacks import ModelCheckpoint
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy, categorical_crossentropy
from keras.models import Model, load_model
from keras.layers import ZeroPadding2D

def create_model(num_classes, opt):
    input_layer = Input((128,128,3))

    # convolutional layers
    conv_layer1 = Conv2D(filters=64, kernel_size=(5, 5),padding='same', strides=2, activation='relu')(input_layer)

    batch1 = BatchNormalization()(conv_layer1)
    # add max pooling to obtain the most imformatic features
    pooling_layer1 = MaxPool2D(pool_size=(2, 2))(batch1)

    conv_layer3 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=1, activation='relu')(pooling_layer1)
    batch2 = BatchNormalization()(conv_layer3)
    pooling_layer2 = MaxPool2D(pool_size=(2, 2))(batch2)

    conv_layer4 = Conv2D(filters=128, kernel_size=(5, 5), padding='same', strides=1, activation='relu')(pooling_layer2)
    batch3 = BatchNormalization()(conv_layer4)
    pooling_layer3 = MaxPool2D(pool_size=(2, 2))(batch3)

    flatten_layer = Flatten()(pooling_layer3)

    dense_layer1 = Dense(units=128, activation='relu')(flatten_layer)

    output_layer = Dense(units=num_classes, activation='sigmoid')(dense_layer1)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def check_accuracy(preds, filenames):
    acc = 0
    test_data = {}
    for path in os.walk('D:\Aditi\plantdisease\PlantVillage\Tomato\\test'):
        tag = path[0].split('\\')[-1]
        c = 0
        for _ in path[2]:
            test_data[_] = tag
    for p in range(len(preds)):
        if CLASS_NAMES[preds[p].argmax()] == test_data[filenames[p].split('\\')[1]]:
            acc += 1
        print(acc, filenames[p], preds[p].argmax())
    return acc / len(preds)


# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20,20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()


data_dir = pathlib.Path('train')
data_dir_val = pathlib.Path('val')
data_dir_test = pathlib.Path('test')
image_count = len(list(data_dir.glob('*/*.JPG')))
CLASS_NAMES = np.array([i.name for i in data_dir.glob('*')])
print(data_dir, CLASS_NAMES)

image_generator = ImageDataGenerator(rescale=1./255, horizontal_flip = True, rotation_range = 360, zoom_range = 0.5)
image_generator_test = ImageDataGenerator(rescale=1./255)
BATCH_SIZE = 10
IMG_HEIGHT = 128
IMG_WIDTH = 128
STEPS_PER_EPOCH = np.ceil(image_count/BATCH_SIZE)
train_data_gen = image_generator.flow_from_directory(directory=str(data_dir), batch_size=BATCH_SIZE, shuffle=True, target_size=(IMG_HEIGHT, IMG_WIDTH))
val_data_gen = image_generator_test.flow_from_directory(directory=str(data_dir_val), batch_size=BATCH_SIZE, shuffle=False, target_size=(IMG_HEIGHT, IMG_WIDTH))
test_data_gen = image_generator_test.flow_from_directory(directory=str(data_dir_test), batch_size=BATCH_SIZE, shuffle=False, target_size=(IMG_HEIGHT, IMG_WIDTH))
# augmented_images = [train_data_gen[0][0][0] for i in range(5)]
# plotImages(augmented_images)

epochs = 200
epochs_range = range(epochs)
c = 11
op = [optimizers.Adagrad(lr = 0.01)]
# class_weight = {'0':0.04, '1':0.1, '2':0.3, '3':0.3, '4':0.26}
story = []

with open('results2.txt', 'a') as file:
    for o in op:
        s = {}
        c += 1
        classifier = create_model(10, o)
        print(classifier.summary())
        f = str(c) + '.h5'
        # callback = ModelCheckpoint(f, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch', **kwargs)
        history = classifier.fit_generator(train_data_gen, steps_per_epoch = STEPS_PER_EPOCH, epochs = epochs, validation_data = val_data_gen)
        classifier.save(f)
        # pred = classifier.predict_generator(test_data_gen)
        # predicted_class_indices = np.argmax(pred, axis = 1)
        # labels = (train_data_gen.class_indices)
        # labels = dict((v,k) for k,v in labels.items())
        # predictions = [labels[k] for k in predicted_class_indices]
        # print(predictions)
        # with open('predictions1.csv', 'w', newline = '') as f:
        #     writer = csv.writer(f)
        #     for _ in predictions:
        #         writer.writerow(_)

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        preds = classifier.predict(test_data_gen)
        print(test_data_gen, preds)
        act_acc = check_accuracy(preds, test_data_gen.filenames)

        s['optimizer'] = str(o)
        s['loss'] = str(loss)
        s['accuracy'] = str(acc)
        s['val_loss'] = str(val_loss)
        s['val_accuracy'] = str(val_acc)
        s['act_acc'] = str(act_acc)
        story.append(s)
        print(s)
        file.write(json.dumps(s))

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy ')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy ')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(str(c) + '.png')

print(story)
