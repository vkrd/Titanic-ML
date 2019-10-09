from data_loader import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# get data
X, y = load_train_data()

z = load_test_data()

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=16))

# Plotting function
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
    plt.ylim([0,20])
    plt.legend()
    plt.show()

    plot_history(history)

# Create model
model = keras.Sequential([
    keras.layers.Dense(8, input_shape=(8, )),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=['accuracy'])


history = model.fit(x=X, y=y, epochs=1500, validation_split=0.2, batch_size=32, verbose=2)

# Plot metrics
plot_history(history)

# Save model output
save_predictions(model.predict(z), "NN.csv")


model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

