import utils
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

raw_df = utils.download()

# compute daily returns and 20 day moving historical volatility
trading_days = 20
df, X = utils.process(df=raw_df, trading_days=trading_days)
utils.test(X=X)
x, y = utils.convert2matrix(data_arr=X, look_back=20)
print(x.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
X_train = np.expand_dims(X_train, 2)
X_test = np.expand_dims(X_test, 2)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Sequential()
# Input layer
model.add(GRU(10, input_shape=(X_train.shape[1], 1),return_sequences=True))
model.add(Dropout(0.2)) 
# Hidden layer
model.add(GRU(10)) 
model.add(Dropout(0.2))
model.add(Dense(units = 1))
model.compile(optimizer="adam",loss="mse",metrics=["mse"])
early_stop = EarlyStopping(monitor = 'val_mse',
                                               patience = 100)
history = model.fit(
    X_train,
    y_train,
    epochs=1200,
    batch_size=32,
    validation_data=(X_test, y_test),
    validation_split = 0.2,
    verbose=1,
    shuffle=False,
    callbacks = [early_stop]
)

mse = history.history['mse']
val_mse = history.history['val_mse']
loss = history.history['loss']
val_loss = history.history['val_loss']

pred = model.predict(X_test)

epochs = range(len(mse))

plt.plot(epochs, mse, 'bo', label='Training mse')
plt.plot(epochs, val_mse, 'b', label='Validation mse')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

plt.figure(figsize=(20, 5), dpi=300)
plt.plot(y_test, label="True Value")
plt.plot(pred, linestyle="-", label="Predicted")
plt.legend()
plt.show()
