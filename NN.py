from data_loader import *
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras

# get data
X, y = load_train_data()

z = load_test_data()

# Create model
model = keras.Sequential([
    keras.layers.Dense(11, input_shape=(8, )),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
              loss='binary_crossentropy',
              metrics=['accuracy'])


history = model.fit(x=X, y=y, epochs=1000, validation_split=0.2, batch_size=32)


# plot metrics
pyplot.plot(history.history['val_loss'])
pyplot.plot(history.history['loss'])
pyplot.show()


model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

