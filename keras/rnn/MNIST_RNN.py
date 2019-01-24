import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Activation,Dense,Dropout,Flatten,Softmax,Convolution2D,MaxPooling2D,Conv2D
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
(x_train,y_train),(x_test,y_test) = mnist.load_data("aa/")
# x_train = x_train.reshape(x_train.shape[0],28,28,1).astype("float32")
# x_test = x_test.reshape(x_test.shape[0],28,28,1).astype("float32")
# x_test/=1000
# x_train/=1000
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(3,3),padding='same',input_shape=(28,28,1),activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=36,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=60,kernel_size=(3,3),padding='same',activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))
# print(model.summary())
# model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
# train_history = model.fit(x=x_train,y=y_train,epochs=1, batch_size=2000, verbose=1)#,validation_split=0.2加上这个参数   训练的结果可能会好一点
# score = model.evaluate(x=x_test,y=y_test,verbose=1)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# model.save("mnist_model")