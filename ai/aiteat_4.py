from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
np.random.seed(10)
from keras.models import Sequential
from keras.layers import Dense


(train_feature,train_label),(test_feature,test_label)=mnist.load_data()
train_feature_vector=train_feature.reshape(len(train_feature),784).astype('float32')
test_feature_vector=test_feature.reshape(len(test_feature),784).astype('float32')
train_feature_normalize=train_feature_vector/255
test_feature_normalize=test_feature_vector/255
train_label_onehot= to_categorical(train_label)
test_label_onehot= to_categorical(test_label)

model=Sequential()
model = Sequential()
model.add(Dense(units=512, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=256, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=128, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=10, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(x=train_feature_normalize,y=train_label_onehot,validation_split=0.4,epochs=100,batch_size=200,verbose=2)
scores=model.evaluate(test_feature_normalize,test_label_onehot)

print('/n 準確率=',scores[1])
model.save('Mnist_mlp_model.h5')
print("Mnist_mlp_model.h5 模型儲存完畢")