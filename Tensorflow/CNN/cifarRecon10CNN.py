import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.layers import Conv2D, Activation, AveragePooling2D, BatchNormalization, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

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

def separarCanales(arreglo):
    res = np.ndarray(shape=(arreglo.shape[0],32,32,3), dtype=float)
    for i in range(arreglo.shape[0]):
        for x in range(0,32):
            res[i,x,:,0] = arreglo[i,(x*32):(x*32)+32]
            res[i,x,:,1] = arreglo[i,(x*32)+1024:(x*32)+1056]
            res[i,x,:,2] = arreglo[i,(x*32)+2048:(x*32)+2080]
        print("# de ejemplo: ", i)
    
    return res

x_train, y_train = cargarDataset()
x_test, y_test = cargarDatasetTest()

x_train = np.reshape(x_train, (50000, 32, 32, 3), order='C')
x_test = np.reshape(x_test, (10000, 32, 32, 3), order='C')

#x_train = separarCanales(x_train)
#x_test = separarCanales(x_test)

#print(np.allclose(x_train,x_train_2))

#Se omite la cantidad de ejemplos, para eso esta el placeholder de None
inputs = tf.keras.Input(shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3]))
x = Conv2D(filters = 32, kernel_size = 6, padding = 'same')(inputs)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size = 5, padding = 'same', strides = 1)(x)

x = Conv2D(filters = 32, kernel_size = 6, padding = 'valid')(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size = 5, padding = 'same', strides = 1)(x)

x = Conv2D(filters = 16, kernel_size = 6, padding = 'valid')(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size = 3, padding = 'same', strides = 1)(x)

x = Conv2D(filters = 8, kernel_size = 3, padding = 'valid')(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size = 3, padding = 'same', strides = 1)(x)

x = Conv2D(filters = 8, kernel_size = 3, padding = 'valid')(x)
x = Activation('relu')(x)
x = AveragePooling2D(pool_size = 3, padding = 'same', strides = 1)(x)

x = Flatten()(x)
x = Dense(10, activation='softmax', kernel_initializer = glorot_uniform(seed=0))(x)

# Creamos el modelo
model = Model(inputs = inputs, outputs = x)
print(model.summary())

#Compilamos el modelo
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01,
                                                momentum=0.0,
                                                nesterov=True,),
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"])

print("\n*Entrenando modelo*\n")
history = model.fit(
    x_train,
    y_train,
    batch_size = 64,
    epochs = 13,
    validation_split=0.2
)

print("\nHistory: \n",history.history)

# Evaluate the model on the test data using `evaluate`
print("\n*Evaluacion del modelo*\n")
results = model.evaluate(x_test, y_test, batch_size=64)
print("Loos en Test, Precision en Test:", results)