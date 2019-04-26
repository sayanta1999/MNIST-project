# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 17:31:38 2019

@author: KIIT
"""

import os
import os.path as path
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib
from keras import backend as K
from keras.datasets import mnist
import numpy as np
seed=9
np.random.seed(seed)
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D

MODEL_NAME='Handwritten Digit Recogniser'

def read_data():
    (x_train,y_train),(x_test,y_test)=mnist.load_data()
    x_train=x_train.reshape(x_train.shape[0],28,28,1)
    x_test=x_test.reshape(x_test.shape[0],28,28,1)
    x_train=x_train.astype('float32')
    x_test=x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train=to_categorical(y_train,10)
    y_test=to_categorical(y_test,10)
    return x_train,x_test,y_train,y_test
    
def build_model():
    model=Sequential()
    model.add(Conv2D(256, kernel_size=(3, 3),activation='relu',input_shape=[28,28,1]))
    # 28*28*64
    model.add(MaxPooling2D(pool_size=2, strides=2, padding='same'))
    model.add(Conv2D(128,(3,3),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(1000,init='uniform',activation='relu'))
    model.add(Dense(10,activation='softmax'))
    return model

def train_model(model,x_train,x_test,y_train,y_test):
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(x_train,y_train,nb_epoch=5,batch_size=120,verbose=1,validation_data=(x_test,y_test))

def export_model(saver, model, input_node_names, output_node_name):
    tf.train.write_graph(K.get_session().graph_def, 'out', \
        MODEL_NAME + '_graph.pbtxt')

    saver.save(K.get_session(), 'out/' + MODEL_NAME + '.chkp')

    freeze_graph.freeze_graph('out/' + MODEL_NAME + '_graph.pbtxt', None, \
        False, 'out/' + MODEL_NAME + '.chkp', output_node_name, \
        "save/restore_all", "save/Const:0", \
        'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_names], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")

def main():
    if not path.exists('out'):
        os.mkdir('out')
    x_train,x_test,y_train,y_test=read_data()
    model=build_model()
    train_model(model,x_train,x_test,y_train,y_test)
    export_model(tf.train.Saver(), model,"conv2d_1_input","dense_2/Softmax")
    
    #Converting to HDF5 format
    graph_def_file = "out/frozen_Handwritten Digit Recogniser.pb"
    input_array = ["conv2d_1_input"]
    output_array = ["dense_2/Softmax"]
    
    #Convert to TensorFlow Lite Model
    converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file,input_array,output_array)
    tflite_model=converter.convert()
    open('HDModel.tflite','wb').write(tflite_model)

if __name__ == '__main__':
    main()