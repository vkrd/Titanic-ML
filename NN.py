from __future__ import absolute_import, division, print_function, unicode_literals
from data_loader import *
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# get data
X, y = load_train_data()

z = load_test_data()


# Plotting function
def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Binary Crossentropy Loss')
    plt.plot(hist['epoch'], hist['binary_crossentropy'],
             label='Train Loss')
    plt.plot(hist['epoch'], hist['val_binary_crossentropy'],
             label='Val Loss')
    #plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Hinge Loss')
    plt.plot(hist['epoch'], hist['hinge'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_hinge'],
             label='Val Error')
    #plt.ylim([0, 20])
    plt.legend()
    plt.show()

# Create model

model = keras.Sequential([
    keras.layers.Dense(8, input_shape=(8,)),
    keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_crossentropy', 'hinge'])

#early_stop = keras.callbacks.EarlyStopping(monitor='binary_crossentropy', patience=20)

history = model.fit(x=X, y=y, epochs=300, validation_split=0.2, batch_size=256, verbose=2)#, callbacks=[early_stop])

# Save model output
print(np.round(model.predict(z).flatten()[:20]))
save_predictions(np.round(model.predict(z).flatten()), "NN.csv")

# Plot metrics
plot_history(history)

model.save_weights('model_weights.h5')

# Save the model architecture
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())
