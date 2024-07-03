from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l1
import tensorflow as tf
import echidna as data
import boundary

X_train = data.X_train
X_validation = data.X_validation
Y_train = to_categorical(data.Y_train)
Y_validation = to_categorical(data.Y_validation)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

model = Sequential()
model.add(Dense(100, activation='sigmoid', activity_regularizer=l1(0.0004)))
model.add(Dense(30, activation='sigmoid', activity_regularizer=l1(0.0004)))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(learning_rate=0.002
              metrics=['accuracy']))

model.fit(X_train, Y_train, validation_data=(X_validation, Y_validation), epochs=30000, batch_size=25)

# Display the descision boundary
boundary.show(model, data.X_train, data.Y_train)
boundary.show(model, data.X_validation, data.Y_validation)