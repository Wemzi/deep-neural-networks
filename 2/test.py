import numpy as np
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.layers
import tensorflow.keras.activations
import tensorflow.keras.callbacks
import matplotlib.pyplot as plt


def preprocess_dataset(data_text):
    lines = data_text.split("\n")
    words = [line.split(";")[:-1] for line in lines]
    attr_names = words[0][:-1]
    vals = words[1:][:-1]
    vals = [[float(item) for item in rec] for rec in vals]
    vals = np.array(vals, dtype=np.float32)
    return vals, attr_names
 
def preprocess_labels(data_text):
    lines = data_text.split("\n")
    words = [line.split(";")[-1] for line in lines]
    attr_name = words[0]
    vals = words[1:-1]
    vals = [float(item) for item in vals]
    vals = np.array(vals, dtype=np.float32)
    return vals, attr_names
features , attr_names = preprocess_dataset(content)  
labels, label_names = preprocess_labels(content)

seed = np.random.randint(0,10000)
np.random.seed(seed)
np.random.shuffle(features)
np.random.seed(seed)
np.random.shuffle(labels)
x_unnorm_train = np.array_split(features,2)[0]
y_train=np.array_split(labels,2)[0]
x_unnorm_val=np.array_split(features,4)[2]
y_val = np.array_split(labels,4)[2]
x_unnorm_test = np.array_split(features,4)[3]
y_test = np.array_split(labels,4)[3]


# implement your solution BELOW
def create_regression_model():
    tf.random.set_seed(42)
    # implement your solution BELOW
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(50, activation="relu", input_dim=x_train.shape[1]))
    model.add(tf.keras.layers.Dropout(0.3, input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.Dense(1, activation="linear", input_dim=x_train.shape[1]))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mse", metrics=['mse','mae'])
    # implement your solution ABOVE
    return model

reg_model = create_regression_model()


reg_model.summary()
history = reg_model.fit(x_unnorm_train, y_train,validation_data=(x_val,y_val), epochs=75)
test_mse = history.history["mse"][-1]
test_mae = history.history["mae"][-1]
plt.plot(history.history['mse'], label="Mean Squared Error")
plt.plot(history.history['mae'], label="Mean Absolute Error")
plt.show()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.show()

A = np.mean(x_unnorm_train,axis=0) 
B = np.std(x_unnorm_train,axis=0)
x_train = (x_unnorm_train-A) / (B)
x_val = (x_unnorm_val-A) / (B)
x_test = (x_unnorm_test-A) / (B)





prec =tf.keras.metrics.Precision()
rec = tf.keras.metrics.Recall()

def create_cl_model():
  cl_model = tf.keras.models.Sequential()
  cl_model.add(tf.keras.layers.Dense(50, activation="tanh", input_dim=x_train.shape[1]))
  cl_model.add(tf.keras.layers.Dense(30, activation="relu",))
  cl_model.add(tf.keras.layers.Dense(4, activation="softmax"))
  cl_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1
                                                     ), loss="categorical_crossentropy", metrics=[prec,'accuracy',rec],)

  return cl_model

cl_model = create_cl_model()