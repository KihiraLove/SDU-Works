import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

keras = tf.keras
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
plt.imshow(x_train[0])

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer='adam',metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_accuracy = model.evaluate(x_test, y_test)

model.save('num_reader.model')
new_model = tf.keras.models.load_model('num_reader.model')
predictions = new_model.predict(x_test)

