# %%
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
assert X_train.shape == (60000, 28, 28)
assert X_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
X_train = X_train / 255.0
X_test = X_test / 255.0

# %%
import matplotlib.pyplot as plt
plt.imshow(X_train[2137], cmap="binary")
plt.axis('off')
plt.show()

# %%
class_names = ["koszulka", "spodnie", "pulower", "sukienka", "kurtka",
"sandał", "koszula", "but", "torba", "kozak"]
class_names[y_train[2137]]

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
model = Sequential([
    Flatten(input_shape=(28, 28)), 
    Dense(300, activation='relu'),  
    Dense(100, activation='relu'), 
    Dense(10, activation='softmax') 
])

# %%
model.summary()
tf.keras.utils.plot_model(model, "fashion_mnist.png", show_shapes=True)

# %%
from keras.optimizers import SGD
from keras.metrics import Accuracy
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# %%


# %%
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
def get_log_dir(x):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(x, current_time)
    return log_dir
tensorboard_callback = TensorBoard(log_dir=get_log_dir("image_logs"), histogram_freq=1)

history = model.fit(X_train, y_train, epochs=20, validation_split=0.1, callbacks=[tensorboard_callback])

# %%
import numpy as np
image_index = np.random.randint(len(X_test))
image = np.array([X_test[image_index]])
confidences = model.predict(image)
confidence = np.max(confidences[0])
prediction = np.argmax(confidences[0])
print("Prediction:", class_names[prediction])
print("Confidence:", confidence)
print("Truth:", class_names[y_test[image_index]])
plt.imshow(image[0], cmap="binary")
plt.axis('off')
plt.show()

# %%
model.save('fashion_clf.keras')

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
housing = fetch_california_housing()

X,y =housing.data,housing.target
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# %%
from tensorflow.keras.layers import  Normalization
normalizer = Normalization(input_shape=X_train.shape[1:])
normalizer.adapt(X_train)
model1 = Sequential([
    normalizer, 
    Dense(50, activation='relu'),  
    Dense(50, activation='relu'), 
    Dense(50, activation='relu'), 
    Dense(1)  
])

# %%

from tensorflow.keras.metrics import RootMeanSquaredError
model1.compile(optimizer='Adam',
              loss='mean_squared_error',
              metrics=[RootMeanSquaredError()])


# %%


# %%
from tensorflow.keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01, verbose=1)
tensorboard_callback = TensorBoard(log_dir=get_log_dir("housing_logs"), histogram_freq=1)

# %%
history = model1.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), 
                    callbacks=[early_stopping_callback, tensorboard_callback])

# %%
model_2 = Sequential([
    normalizer,
    Dense(100, activation='relu'),
    Dense(100, activation='relu'),
    Dense(1)
])


model_2.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])


tensorboard_callback_2 = TensorBoard(log_dir=os.path.join(get_log_dir("housing_logs_2")), histogram_freq=1)


history_2 = model_2.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), 
                        callbacks=[early_stopping_callback, tensorboard_callback_2])
model_2.save('reg_housing_2.keras')

# %%
model_3 = Sequential([
    normalizer,
    Dense(200, activation='relu'),
    Dense(200, activation='relu'),
    Dense(1)
])


model_3.compile(optimizer='adam', loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])

tensorboard_callback_3 = TensorBoard(log_dir=os.path.join(get_log_dir("housing_logs_3")), histogram_freq=1)

history_3 = model_3.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val), 
                        callbacks=[early_stopping_callback, tensorboard_callback_3])

model_3.save('reg_housing_3.keras')

# %%
model1.save("reg_housing_1.keras")


