
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import utils

## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# # URL of the white wine dataset
URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv'

# # load the dataset from the URL
white_df = pd.read_csv(URL, sep=";")

# # fill the `is_red` column with zeros.
white_df["is_red"] = 0# YOUR CODE HERE

# # keep only the first of duplicate items
white_df = white_df.drop_duplicates(keep='first')

# You can click `File -> Open` in the menu above and open the `utils.py` file
# in case you want to inspect the unit tests being used for each graded function.

utils.test_white_df(white_df)
print(white_df.alcohol[0])
print(white_df.alcohol[100])

# EXPECTED OUTPUT
# 8.8
# 9.1

## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# # URL of the red wine dataset
URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# # load the dataset from the URL
red_df = pd.read_csv(URL, sep=";")

# # fill the `is_red` column with ones.
red_df["is_red"] = 1 # YOUR CODE HERE

# # keep only the first of duplicate items
red_df = red_df.drop_duplicates(keep='first')

utils.test_red_df(red_df)

print(red_df.alcohol[0])
print(red_df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 10.2

df = pd.concat([red_df, white_df], ignore_index=True)
print(df.alcohol[0])
print(df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 9.5

# NOTE: In a real-world scenario, you should shuffle the data.
# YOU ARE NOT going to do that here because we want to test
# with deterministic data. But if you want the code to do it,
# it's in the commented line below:

#df = df.iloc[np.random.permutation(len(df))]

df['quality'].hist(bins=20);
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# # get data with wine quality greater than 4 and less than 8
df = df[(df['quality'] > 4) & (df['quality']<8)]# YOUR CODE HERE) & (df['quality'] < YOUR CODE HERE )]

# # reset index and drop the old one
df = df.reset_index(drop=True)

utils.test_df_drop(df)

print(df.alcohol[0])
print(df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 10.9
df['quality'].hist(bins=20);

## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.


## Please do not change the random_state parameter. This is needed for grading.

# # split df into 80:20 train and test sets
train, test = train_test_split(df, test_size=0.2, random_state=1)  # YOUR CODE HERE, random_state = 1)

# # split train into 80:20 train and val sets
train, val = train_test_split(train, test_size=0.2, random_state=1)  # YOUR CODE HERE, random_state = 1)

utils.test_data_sizes(train.size, test.size, val.size)

train_stats = train.describe()
train_stats.pop('is_red')
train_stats.pop('quality')
train_stats = train_stats.transpose()

train_stats

def format_output(data):
    is_red = data.pop('is_red')
    is_red = np.array(is_red)
    quality = data.pop('quality')
    quality = np.array(quality)
    return (quality, is_red)


## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.


# # format the output of the train set
train_Y = format_output(train)  # YOUR CODE HERE)

# # format the output of the val set
val_Y = format_output(val)  # YOUR CODE HERE)

# # format the output of the test set
test_Y = format_output(test)  # YOUR CODE HERE)

utils.test_format_output(df, train_Y, val_Y, test_Y)

train.head()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.


# # normalize the train set
norm_train_X = norm(train)  # YOUR CODE HERE)

# # normalize the val set
norm_val_X = norm(val)  # YOUR CODE HERE)

# # normalize the test set
norm_test_X = norm(test)  # YOUR CODE HERE)

utils.test_norm(norm_train_X, norm_val_X, norm_test_X, train, val, test)


## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.


def base_model(inputs):
    # connect a Dense layer with 128 neurons and a relu activation
    x = Dense(128, activation='relu', name="first_dense_layer")(inputs)  # YOUR CODE HERE
    # connect another Dense layer with 128 neurons and a relu activation
    x = Dense(128, activation='relu', name="second_dense_layer")(x)  # YOUR CODE HERE
    return x

utils.test_base_model(base_model)


## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.


def final_model(inputs):
    # get the base model
    x = base_model(inputs)

    # connect the output Dense layer for regression
    wine_quality = Dense(units='1', name='wine_quality')(x)

    # connect the output Dense layer for classification. this will use a sigmoid activation.
    wine_type = Dense(units='1', activation='sigmoid', name='wine_type')(x)  # YOUR CODE HERE, name='wine_type')(x)

    # define the model using the input and output layers
    model = Model(inputs=inputs, outputs=[wine_quality, wine_type])  # YOUR CODE HERE, outputs=# YOUR CODE HERE)

    return model

utils.test_final_model(final_model)

## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.


inputs = tf.keras.layers.Input(shape=(11,))
rms = tf.keras.optimizers.RMSprop(lr=0.0001)
model = final_model(inputs)

model.compile(optimizer=rms,
              loss = {'wine_type':"binary_crossentropy",# YOUR CODE HERE,
                      'wine_quality':tf.keras.losses.MSE # YOUR CODE HERE
                     },
              metrics = {'wine_type':"accuracy", # YOUR CODE HERE,
                         'wine_quality':tf.keras.metrics.RootMeanSquaredError()# YOUR CODE HERE
                       }
             )

utils.test_model_compile(model)
## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



history = model.fit(norm_train_X, # YOUR CODE HERE,
                    train_Y, # YOUR CODE HERE,
                    epochs = 180, validation_data=(norm_val_X, val_Y))# YOUR CODE HERE, # YOUR CODE HERE))

utils.test_history(history)
# Gather the training metrics
loss, wine_quality_loss, wine_type_loss, wine_quality_rmse, wine_type_accuracy = model.evaluate(x=norm_val_X, y=val_Y)

print()
print(f'loss: {loss}')
print(f'wine_quality_loss: {wine_quality_loss}')
print(f'wine_type_loss: {wine_type_loss}')
print(f'wine_quality_rmse: {wine_quality_rmse}')
print(f'wine_type_accuracy: {wine_type_accuracy}')

# EXPECTED VALUES
# ~ 0.30 - 0.38
# ~ 0.30 - 0.38
# ~ 0.018 - 0.030
# ~ 0.50 - 0.62
# ~ 0.97 - 1.0

# Example:
#0.3657050132751465
#0.3463745415210724
#0.019330406561493874
#0.5885359048843384
#0.9974651336669922

predictions = model.predict(norm_test_X)
quality_pred = predictions[0]
type_pred = predictions[1]
print(quality_pred[0])

# EXPECTED OUTPUT
# 5.6 - 6.0
print(type_pred[0])
print(type_pred[944])

# EXPECTED OUTPUT
# A number close to zero
# A number close to or equal to 1

def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)

def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="black" if cm[i, j] > thresh else "white")
    plt.show()

def plot_diff(y_true, y_pred, title = '' ):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100, 100], [-100, 100])
    return plt

plot_metrics('wine_quality_root_mean_squared_error', 'RMSE', ylim=2)
plot_metrics('wine_type_loss', 'Wine Type Loss', ylim=0.2)

plot_confusion_matrix(test_Y[1], np.round(type_pred), title='Wine Type', labels = [0, 1])
scatter_plot = plot_diff(test_Y[0], quality_pred, title='Type')


