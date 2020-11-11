
#genre classifier
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib as mpl
import matplotlib.pyplot as plt


dataset_path = "/content/drive/My Drive/data.json"
def load_data(dataset_path):
  with open(dataset_path,"r") as fp:
    data = json.load(fp)
  x = np.array(data["mfcc"])
  y = np.array(data["labels"])

  print("Data successfully loaded")

  return x, y

 
if __name__ == "__main__":

#load the data from the json
  x, y = load_data(dataset_path)
#split the data into training and the test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
#Build the network architecture
model = keras.Sequential([
                          
                          keras.layers.Flatten(input_shape = (x.shape[1], x.shape[2])),
                          keras.layers.Dense(512, activation = 'relu', kernel_regularizer= keras.regularizers.l2(0.001)),
                          keras.layers.Dropout(0.3),
                          keras.layers.Dense(256, activation = 'relu', kernel_regularizer= keras.regularizers.l2(0.001)),
                           keras.layers.Dropout(0.3),
                          keras.layers.Dense(64, activation = 'relu', kernel_regularizer= keras.regularizers.l2(0.001)),
                           keras.layers.Dropout(0.3),
                          keras.layers.Dense(10, activation = 'softmax')
])
#Compile the network
optimiser = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimiser,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model.summary()
#Train the network
history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
                     batch_size=25, epochs=100)

def plot_history(history):
    
    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()
