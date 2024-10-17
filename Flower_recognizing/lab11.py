# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization,Rescaling,InputLayer,GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pickle

# %%
import tensorflow_datasets as tfds
[test_set_raw, valid_set_raw, train_set_raw], info = tfds.load(
"tf_flowers",
split=['train[:10%]', "train[10%:25%]", "train[25%:]"],
as_supervised=True,
with_info=True)

# %%
class_names = info.features["label"].names
n_classes = info.features["label"].num_classes
dataset_size = info.splits["train"].num_examples

# %%
plt.figure(figsize=(12, 8))
index = 0
sample_images = train_set_raw.take(9)
for image, label in sample_images:
    index += 1
    plt.subplot(3, 3, index)
    plt.imshow(image)
    plt.title("Class: {}".format(class_names[label]))
    plt.axis("off")
plt.show(block=False)

# %%
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    return resized_image, label
batch_size = 32
train_set = train_set_raw.map(preprocess).shuffle(dataset_size).batch(batch_size).prefetch(1)
valid_set = valid_set_raw.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set_raw.map(preprocess).batch(batch_size).prefetch(1)

# %%
plt.figure(figsize=(8, 8))
sample_batch = train_set.take(1)
print(sample_batch)
for X_batch, y_batch in sample_batch:
    for index in range(12):
        plt.subplot(3, 4, index + 1)
        plt.imshow(X_batch[index]/255.0)
        plt.title("Class: {}".format(class_names[y_batch[index]]))
        plt.axis("off")
plt.show()

# %%
normalization_layer = Rescaling(scale=1./255 )

# %%
model = Sequential([

    InputLayer(input_shape=(224, 224, 3)),
    Conv2D(32, (7, 7), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

# %%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
history = model.fit(train_set, epochs=10, validation_data=valid_set)

# %%
acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]
results = (acc_train, acc_valid, acc_test)

# %%
with open('simple_cnn_acc.pkl','wb') as f:
    pickle.dump(results,f)
model.save('simple_cnn_flowers.keras')

# %%
from tensorflow.keras.applications import xception


# %%
def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = tf.keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

# %%
base_model = tf.keras.applications.Xception(
    weights="imagenet",
    include_top=False
)

# %%
tf.keras.utils.plot_model(base_model, show_shapes=True)

# %%
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
outputs = Dense(n_classes, activation='softmax')(x)

# %%
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
history = model.fit(train_set, epochs=5, validation_data=valid_set)

# %%
for layer in base_model.layers:
    layer.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%
history_fine = model.fit(train_set, epochs=3, validation_data=valid_set)

# %%
acc_train = model.evaluate(train_set)[1]
acc_valid = model.evaluate(valid_set)[1]
acc_test = model.evaluate(test_set)[1]

results = (acc_train, acc_valid, acc_test)

# %%
with open('xception_acc.pkl','wb') as f:
    pickle.dump(results,f)
model.save('xception_flowers.keras')


