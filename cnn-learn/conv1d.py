from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Flatten
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D 
import numpy as np

sample_size=1000
x_train = np.random.randint( 1, size=(sample_size,10000,1) )
y_train = np.random.randint( 1, size=(sample_size,4000) )

model = Sequential()
model.add(Conv1D(64, 9, activation='relu', padding="same", input_shape=(10000,1)))
model.add(Conv1D(64, 9, activation='relu', padding="same"))
model.add(Conv1D(64, 9, activation='relu', padding="same"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(4000,name="full_connect_layer"))
model.add(Dropout(0.5))
model.add(Activation("sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 打印 fc 层的输出
#output = keras.Model(inputs=model.input, outputs=model.get_layer('full_connect_layer').output).predict(data_1d)
#print(output)

# 打印网络结构
print(model.summary())

model.fit(x_train, y_train, batch_size=16, epochs=10)
#score = model.evaluate(x_test, y_test, batch_size=16)