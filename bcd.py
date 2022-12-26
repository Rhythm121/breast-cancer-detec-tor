import numpy as ny
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split
 #loading the data dfrom the data set
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
print(breast_cancer_dataset)
#loading the data from data frame
data_frame=pd.DataFrame(breast_cancer_dataset,columns=breast_cancer_dataset.feature_names)
data_frame.head()
#adding target column to data frame
data_frame['label']=breast_cancer_dataset.target
#print last 5 rows of dataframe
data_frame.tail()
#number of rows and column
data_frame.shape
data_frame.info()

data_frame.describe()
#checking the distribution on traget value
data_frame['label'].value_counts()
data_frame.groupby('label').mean()
data_frame.groupby('label').mean()
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_std = scaler.fit_transform(X_train)

X_test_std = scaler.transform(X_test)
# importing tensorflow and Keras
import tensorflow as tf 
tf.random.set_seed(3)
from tensorflow import keras
model = keras.Sequential([
keras.layers.Flatten(input_shape=(30,)),
keras.layers.Dense(20, activation='relu'),
keras.layers.Dense(2, activation='sigmoid')
])
history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'lower right')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')

plt.legend(['training data', 'validation data'], loc = 'upper right')
loss, accuracy = model.evaluate(X_test_std, Y_test)
print(accuracy)
print(X_test_std.shape)
print(X_test_std[0])
Y_pred = model.predict(X_test_std)
print(Y_pred.shape)
print(Y_pred[0])
print(X_test_std)
print(Y_pred)
#  argmax function

my_list = [0.25, 0.56]

index_of_max_value = np.argmax(my_list)
print(my_list)
print(index_of_max_value)
Y_pred_labels = [np.argmax(i) for i in Y_pred]
print(Y_pred_labels)
input_data = (11.76,21.6,74.72,427.9,0.08637,0.04966,0.01657,0.01115,0.1495,0.05888,0.4062,1.21,2.635,28.47,0.005857,0.009758,0.01168,0.007445,0.02406,0.001769,12.98,25.72,82.98,516.5,0.1085,0.08615,0.05523,0.03715,0.2433,0.06563)

# change the input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for one data point
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
input_data_std = scaler.transform(input_data_reshaped)

prediction = model.predict(input_data_std)
print(prediction)

prediction_label = [np.argmax(prediction)]
print(prediction_label)

if(prediction_label[0] == 0):
  print('The tumor is Malignant')

else:
  print('The tumor is Benign')

