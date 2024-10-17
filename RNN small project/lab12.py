# %%
import tensorflow as tf
import os
import requests
import zipfile
import pandas as pd
import pickle

# %%
url = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"

zip_path = "bike_sharing_dataset.zip"

extract_path = "datasets"

response = requests.get(url)
with open(zip_path, 'wb') as f:
    f.write(response.content)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

os.remove(zip_path)

# %%
df = pd.read_csv('datasets/hour.csv',
parse_dates={'datetime': ['dteday', 'hr']},
date_format='%Y-%m-%d %H',
index_col='datetime')

# %%
print((df.index.min(), df.index.max()))

# %%
(365 + 366) * 24 - len(df)

# %%
resampled_df = df.resample('H').mean()
resampled_df['casual'] = resampled_df['casual'].fillna(0)
resampled_df['registered'] = resampled_df['registered'].fillna(0)
resampled_df['cnt'] = resampled_df['cnt'].fillna(0)


resampled_df['temp'] = resampled_df['temp'].interpolate()
resampled_df['atemp'] = resampled_df['atemp'].interpolate()
resampled_df['hum'] = resampled_df['hum'].interpolate()
resampled_df['windspeed'] = resampled_df['windspeed'].interpolate()

resampled_df['holiday'] = df['holiday'].resample('H').ffill().fillna(method='ffill')
resampled_df['weekday'] = df['weekday'].resample('H').ffill().fillna(method='ffill')
resampled_df['workingday'] = df['workingday'].resample('H').ffill().fillna(method='ffill')
resampled_df['weathersit'] = df['weathersit'].resample('H').ffill().fillna(method='ffill')

# %%
df = resampled_df.drop(columns=['instant','season','yr','mnth'],axis=1)
df.notna().sum()

# %%
df[['casual', 'registered', 'cnt', 'weathersit']].describe()

# %%
df.casual /= 1e3
df.registered /= 1e3
df.cnt /= 1e3
df.weathersit /= 4

# %%
df_2weeks = df[:24 * 7 * 2]
df_2weeks[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))

# %%
df_daily = df.resample('W').mean()
df_daily[['casual', 'registered', 'cnt', 'temp']].plot(figsize=(10, 3))

# %%
mae_daily = df['cnt'].diff(24).abs().mean() * 1e3
mae_weekly = df['cnt'].diff(24*7).abs().mean() * 1e3
mae_baseline = (mae_daily, mae_weekly)
print(mae_baseline)
with open('mae_baseline.pkl', 'wb') as f:
    pickle.dump(mae_baseline, f)

# %%
cnt_train = df['cnt']['2011-01-01 00:00':'2012-06-30 23:00']
cnt_valid = df['cnt']['2012-07-01 00:00':]

# %%
seq_len = 1 * 24
train_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_train.to_numpy(),
    targets=cnt_train[seq_len:],
    sequence_length=seq_len,
    batch_size=32,
    shuffle=True,
    seed=42
)
valid_ds = tf.keras.utils.timeseries_dataset_from_array(
    cnt_valid.to_numpy(),
    targets=cnt_valid[seq_len:],
    sequence_length=seq_len,
    batch_size=32
)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber
model = tf.keras.Sequential([
tf.keras.layers.Dense(1, input_shape=[seq_len])
])
optimizer = SGD(learning_rate=0.01, momentum=0.9)
loss = Huber()
model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

# %%
history = model.fit(train_ds, epochs=20, batch_size=32, validation_data=(valid_ds))

# %%
model.save('model_linear.keras')

mae_val = model.evaluate(valid_ds)

print(mae_val)

# %%
with open('mae_linear.pkl', 'wb') as f:
    pickle.dump((mae_val,), f)

# %%
model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
optimizer = SGD(learning_rate=0.08, momentum=0.9)
loss = Huber()
model.compile(optimizer=optimizer, loss=loss, metrics=['mae'])

# %%
history = model.fit(train_ds, epochs=20, batch_size=32, validation_data=(valid_ds))

# %%
model.save('model_rnn1.keras')
mae_rnn1 = model.evaluate(valid_ds)
print((mae_rnn1,))
with open('mae_rnn1.pkl', 'wb') as f:
    pickle.dump((mae_rnn1,),f)

# %%
model_rnn32 = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, input_shape=[None, 1])
])
model_rnn32.add(Dense(1))
model_rnn32.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),
                    loss=tf.keras.losses.Huber(),
                    metrics=['mae'])


# %%
history_rnn32 = model_rnn32.fit(train_ds, epochs=20, validation_data=valid_ds)

# %%
model_rnn32.save('model_rnn32.keras')
mae_rnn32 = model_rnn32.evaluate(valid_ds)
with open('mae_rnn32.pkl', 'wb') as f:
    pickle.dump((mae_rnn32,), f)

# %%
model_rnn_multiple = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[None, 1]),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(32, return_sequences=True),
    tf.keras.layers.SimpleRNN(1)  
])

# %%
model_rnn_multiple.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9),
                           loss=tf.keras.losses.Huber(),
                           metrics=['mae'])

# %%
history_rnn_multiple = model_rnn_multiple.fit(train_ds, epochs=20, validation_data=valid_ds)

# %%
model_rnn_multiple.save('model_rnn_deep.keras')

mae_rnn_multiple = model_rnn_multiple.evaluate(valid_ds)
print((mae_rnn_multiple,))
with open('mae_rnn_deep.pkl', 'wb') as f:
    pickle.dump((mae_rnn_multiple,), f)

# %%



