import tensorflow as tf
import pickle
from tensorflow.keras.layers import Dense, Flatten, Conv2D


def cargarDataset():

    x_train = []
    y_train = []
    # open a file, where you stored the pickled data
    db = open('.\Datasets\cifar-10-batches-py\data_batch_1', 'rb')
    datos = pickle.load(db, fix_imports=True, encoding="latin",
                        errors="strict", buffers=None)
    x_train.extend(datos['data']/255)
    y_train.extend(datos['labels'])
    db.close()

    db = open('.\Datasets\cifar-10-batches-py\data_batch_2', 'rb')
    x_train.extend(datos['data']/255)
    y_train.extend(datos['labels'])
    db.close()

    db = open('.\Datasets\cifar-10-batches-py\data_batch_3', 'rb')
    x_train.extend(datos['data']/255)
    y_train.extend(datos['labels'])
    db.close()

    db = open('.\Datasets\cifar-10-batches-py\data_batch_4', 'rb')
    x_train.extend(datos['data']/255)
    y_train.extend(datos['labels'])
    db.close()

    db = open('.\Datasets\cifar-10-batches-py\data_batch_5', 'rb')
    x_train.extend(datos['data']/255)
    y_train.extend(datos['labels'])
    db.close()

    return tf.constant(x_train), tf.constant(y_train)


def cargarDatasetTest():
    x_test = []
    y_test = []
    # open a file, where you stored the pickled data
    db = open('.\Datasets\cifar-10-batches-py\\test_batch', 'rb')
    datos = pickle.load(db, fix_imports=True, encoding="latin",
                        errors="strict", buffers=None)
    x_test.extend(datos['data']/255)
    y_test.extend(datos['labels'])
    db.close()

    return tf.constant(x_test), tf.constant(y_test)


x_train, y_train = cargarDataset()
x_test, y_test = cargarDatasetTest()

# Agrega una dimension de canales
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)).shuffle(10000).batch(64)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(64)

input = tf.keras.Input(shape=(3072,))
dense = (tf.keras.layers.Dense(3072, activation='relu')(input))
dense = (tf.keras.layers.Dense(1536, activation='relu')(dense))
dense = (tf.keras.layers.Dense(768, activation='relu')(dense))
dense = (tf.keras.layers.Dense(255, activation='relu')(dense))
outputs = tf.keras.layers.Dense(10, activation='softmax')(dense)

model = tf.keras.Model(inputs=input, outputs=outputs, name="prueba1")

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=15,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
