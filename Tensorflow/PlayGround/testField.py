import tensorflow as tf

import ImageUtility

#texto = tf.Variable("fasfaa");
#print(texto)
#print(texto.is_bool)

tensor = tf.constant([
    [
        [1,2,3],[4,5,6]
    ],
    [
        [7,8,9],[10,11,12]
    ]
])
# El ultimo arreglo es de izquierda a derecha
# El penultimo es de arriba a abajo
# El antepenultimo de una matriz izquierda y una matriz derecha 
# Y antes de eso pues una matriz arriba y una matriz abajo
print(tf.pad(tensor,[
    [1,1],[1,1],[1,1]
]))

imagePIL = ImageUtility.ImageUtilityPIL("C:/Users/Home/Documents/Proyectos/MLProjects/Tensorflow/YOLO/datasetLocal/")
imagePIL.mostrarImagen("testImage.jpg")