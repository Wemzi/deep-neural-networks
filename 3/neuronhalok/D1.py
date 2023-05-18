import numpy as np
import tensorflow as tf
import tensorflow.keras.models
import tensorflow.keras.optimizers
import tensorflow.keras.layers
import tensorflow.keras.activations
import tensorflow.keras.callbacks
import keras_tuner as kt
print("hello")
import matplotlib.pyplot as plt

#import tester after importing tensorflow, to make sure correct tf version is imported
from tester import Tester

   # BELOW

tester = Tester('IDU27K')
content = tester.get_dataset_content()

print(len(content))
print(content[:500])


   # BELOW
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
 
   # ABOVE
 
tester.test('dataset_shape', features, labels)

   # BELOW
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
   # ABOVE

tester.test('dataset_split', x_unnorm_train, x_unnorm_val, x_unnorm_test,\
                             y_train, y_val, y_test)


   # BELOW

A = np.mean(x_unnorm_train,axis=0) 
B = np.std(x_unnorm_train,axis=0)
x_train = (x_unnorm_train-A) / (B)
x_val = (x_unnorm_val-A) / (B)
x_test = (x_unnorm_test-A) / (B)




   # ABOVE

tester.test('dataset_rescale', x_train, x_val, x_test)


   # BELOW
def create_regression_model():
    tf.random.set_seed(42)
       # BELOW
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(50, activation="relu", input_dim=x_train.shape[1]))
    model.add(tf.keras.layers.Dropout(0.3, input_shape=(x_train.shape[1],)))
    model.add(tf.keras.layers.Dense(1, activation="linear", input_dim=x_train.shape[1]))
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss="mse", metrics=['mse','mae'])
       # ABOVE
    return model

reg_model = create_regression_model()


   # ABOVE

tester.test('reg_model_architecture', reg_model)

   # BELOW
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



   # ABOVE

tester.test('reg_model_learning', test_mse, test_mae)

   # BELOW

categs = np.percentile(y_test,[25,50,75])
c1 = categs[0]
c2 = categs[1]
c3 = categs[2]
def pre_process(data):
  c1tmp = []
  c2tmp = []
  c3tmp = []
  c4tmp = []
  for val in data:
    if(val<=c1):
      c1tmp.append(0)
    elif(val>c1 and val<=c2):
      c2tmp.append(1)
    elif(val>c2 and val<=c3):
      c3tmp.append(2)
    elif(val>c3):
      c4tmp.append(3)
  retarr=c1tmp+c2tmp+c3tmp+c4tmp
  return np.array(retarr, dtype=np.float32)

y_cat_train = pre_process(y_train)
y_cat_val = pre_process(y_val)
y_cat_test = pre_process(y_test)






   # ABOVE

tester.test('cl_dataset', y_cat_train, y_cat_val, y_cat_test)

   # BELOW
nb_classes = 4

def indices_to_one_hot(data, nb_classes):
    targets =np.array(data,dtype=int).reshape(-1)
    return np.array(np.eye(nb_classes)[targets],dtype=np.float32)

y_onehot_train= indices_to_one_hot(y_cat_train,4)
y_onehot_val=indices_to_one_hot(y_cat_val,4)
y_onehot_test=indices_to_one_hot(y_cat_test,4)


   # ABOVE

tester.test('cl_onehot', y_onehot_train, y_onehot_val, y_onehot_test)

   # BELOW

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




   # ABOVE

tester.test('cl_model_architecture', cl_model)

   # BELOW
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15)
history = cl_model.fit(x_train,y_onehot_train,validation_data=(x_val,y_onehot_val),batch_size=128,epochs=75,callbacks=[earlystopping_callback])
test_ce, test_prec, test_acc, test_re = cl_model.evaluate(x_test, y_onehot_test)
test_f1 = 2*((test_prec*test_re)/(test_prec+test_re))

plt.plot(history.history['loss'], label = "training loss")
plt.plot(history.history['accuracy'],label = "training accuracy")
plt.legend()
plt.show()

plt.plot(history.history['val_loss'],label="Validation loss")
plt.plot(history.history['val_accuracy'],label="Validation accuracy")
plt.legend()
plt.show()


   # ABOVE

tester.test('cl_model_learning', test_ce, test_acc, test_f1)
tester.print_all_tests_successful()

#stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
#tuner.search(x_train, y_onehot_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
#best_hps=tuner.get_best_hyperparameters(num_trials=35)[0]

print(best_hps)



   # ABOVE

tester.test('reg_model_learning', test_mse, test_mae)