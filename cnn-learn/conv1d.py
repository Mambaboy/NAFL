#coding=utf-8
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D 
import numpy as np
import keras
from keras.utils import plot_model

sample_size=1000
input_length=1000
output_lentgh=400

x_train = np.random.randint( 1, size=(sample_size,input_length,1) )
y_train = np.random.randint( 1, size=(sample_size, output_lentgh) )
data_1d = np.random.randint( 1, size=(1,input_length,1) )


model = Sequential()
model.add(Conv1D(64, 9, activation='relu', strides=3, padding="same", input_shape=(input_length,1)))
model.add(Conv1D(64, 9, activation='relu', strides=3, padding="same"))
model.add(Conv1D(64, 9, activation='relu', strides=3, padding="same"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense( output_lentgh,name="full_connect_layer"))
model.add(Dropout(0.25))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 打印网络结构
print(model.summary())


model.fit(x_train, y_train, batch_size=16, epochs=10)
#score = model.evaluate(x_test, y_test, batch_size=16)

model.save("./model.h5")
plot_model(model, to_file='model.png')

# 打印 fc 层的输出
print (u"the input is ")
print(data_1d.shape)
output = keras.Model(inputs=model.input, outputs=model.get_layer('full_connect_layer').output).predict(data_1d)
print(u"the result")
print (output.shape)


