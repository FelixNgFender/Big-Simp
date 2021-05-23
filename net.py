import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow .keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.impute import SimpleImputer


data = pd.read_csv('./framingham.csv')
data.head()

data_fix = data.copy()
# fill data
mean_imputer = SimpleImputer(strategy='mean')
data_fix.iloc[:, :] = mean_imputer.fit_transform(data_fix)
y = data_fix["TenYearCHD"]
X = data_fix.drop("TenYearCHD", axis=1)
# t copy trên cái notebook của mình
X = sklearn.preprocessing.normalize(X, norm='l2', axis=1, copy=True, return_norm=False)
y = np.array(y).reshape(-1, 1)  # forrmat để input vào model
# là cái neural net b thg
model = Sequential()
# thêm 1 layer 15 node
model.add(Dense(15, input_dim=15, activation='relu'))
# thêm 1 layer output (có g trị là probability)
model.add(Dense(1, activation='sigmoid'))
# optimizer để GD
opt = tf.keras.optimizers.SGD(learning_rate=0.1)  # SGD : stochastic gradient descent
model.compile(optimizer="adam",  # optimizer bthg của keras
              loss="binary_crossentropy",  # giống cross entropy nhưng cho binary categorization
              metrics=["accuracy"])

# train model = data
history = model.fit(X, y, epochs=13, batch_size=12, validation_split=0.08)

# plot accuracy theo tg
plt.plot(history.history['loss'], label='MAE (training data)')
plt.plot(history.history['val_loss'], label='MAE (validation data)')
plt.title('MAE for <3')
plt.ylabel('MAE value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()
