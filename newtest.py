# %%
# 1. Introduction
# Corrected Code

# %%
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
import itertools

from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from keras.models import *
from keras.layers import *
from keras.optimizers import RMSprop, Adam, SGD
from keras.optimizers import Nadam, Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import vgg16, inception_v3, resnet50
from tensorflow.keras import backend
from keras.callbacks import EarlyStopping

from keras.layers import Average

sns.set(style='white', context='notebook', palette='deep')

# %%
def add_one_to_one_correlation_line(ax, min_factor=1, max_factor=1, **plot_kwargs):
    lim_min, lim_max = pd.DataFrame([ax.get_ylim(), ax.get_xlim()]).agg({0: 'min', 1: 'max'})
    lim_min *= min_factor
    lim_max *= max_factor
    plot_kwargs_internal = dict(color='grey', ls='--')
    plot_kwargs_internal.update(plot_kwargs)
    ax.plot([lim_min, lim_max], [lim_min, lim_max], **plot_kwargs_internal)
    ax.set_ylim([lim_min, lim_max])
    ax.set_xlim([lim_min, lim_max])


# %% [markdown]
# # 2. Data preparation
# ## 2.1 Load and review data

# %%
# Load the data
df = pd.read_csv("Dataset\labels.csv")

# %%
# Map each id to its appropriate file name
df['image_name'] = df['id'].map('seq_{:06d}.jpg'.format)

# %%
df.describe()

# %%
df['count'].hist(bins=30);

# %% [markdown]
# ## 2.2 Setup data generator with optional augmentation 

# %% [markdown]
# In order to avoid overfitting problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit.
# 
# For example, the number is not centered 
# The scale is not the same (some who write with big/small numbers)
# The image is rotated...
# 
# Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques. Some popular augmentations people use are grayscales, horizontal flips, vertical flips, random crops, color jitters, translations, rotations, and much more. 
# 
# The approaches can help avoid overfitting, but it is not clear that we want to add this extra variance in this specific problem. You can play with the optional augmentations below and see how they affect the results.

# %%
# Setup some constants
size = 224
batch_size = 64

# %%
# ImageDataGenerator - with defined augmentaions
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.2,
    preprocessing_function=resnet50.preprocess_input
)

# %% [markdown]
# ## 2.3 Load image data
# We use the defined ImageDataGenerator to read the images using the dataframe we read earlier.

# %%
flow_params = dict(
    dataframe=df,
    directory='Dataset/frames/frames',
    x_col="image_name",
    y_col="count",
    weight_col=None,
    target_size=(size, size),
    color_mode='rgb',
    class_mode="raw",
    batch_size=batch_size,
    shuffle=True,
    seed=0,
)

# The dataset is split to training and validation sets at this point
train_generator = datagen.flow_from_dataframe(
    subset='training',
    **flow_params    
)
valid_generator = datagen.flow_from_dataframe(
    subset='validation',
    **flow_params
)

# %%
batch = next(train_generator)
fig, axes = plt.subplots(4, 4, figsize=(14, 14))
axes = axes.flatten()
for i in range(16):
    ax = axes[i]
    ax.imshow(batch[0][i])
    ax.axis('off')
    ax.set_title(batch[1][i])
plt.tight_layout()
plt.show()

# %% [markdown]
# # 3. CNN
# ## 3.1 Load and modify the pretrained models

# %% [markdown]
# I used the Keras implementation of ResNet50 and Inception V3 - convolutional neural networks for image recognition tasks. They have been pre-trained on the ImageNet dataset.

# %%
base_model_resnet = resnet50.ResNet50(
    weights='imagenet',  # Load the pretrained weights, trained on the ImageNet dataset.
    include_top=False,  # We don't include the fully-connected layer at the top of the network - we need to modify the top.
    input_shape=(size, size, 3),  # 224x224 was the original size ResNet was trained on, so I decided to use this.
    pooling='avg',  # A global average pooling layer will be added after the last convolutional block.
)

base_model_inception = inception_v3.InceptionV3(
    weights='imagenet',
    include_top=False,
    input_shape=(size, size, 3),
    pooling='avg'
)

# %%
# Regularization
dropout_rate = 0.5

# Modify top layers for ResNet50
x_resnet = base_model_resnet.output
x_resnet = Dense(512, activation='relu')(x_resnet)
x_resnet = Dropout(dropout_rate)(x_resnet)
predictions_resnet = Dense(1, activation='linear')(x_resnet)

# Modify top layers for Inception V3
x_inception = base_model_inception.output
x_inception = Dense(512, activation='relu')(x_inception)
x_inception = Dropout(dropout_rate)(x_inception)
predictions_inception = Dense(1, activation='linear')(x_inception)

# %%
# Define the combined model architecture
def combined_model(base_model_resnet, base_model_inception):
    input_resnet = base_model_resnet.input
    input_inception = base_model_inception.input

    output_resnet = base_model_resnet.output
    output_inception = base_model_inception.output

    # Concatenate the outputs of both models
    combined_features = Concatenate()([output_resnet, output_inception])

    # Add dense layers
    x = Dense(512, activation='relu')(combined_features)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='linear')(x)

    # Create the combined model
    model = Model(inputs=[input_resnet, input_inception], outputs=predictions)

    return model

# Create the combined model
combined_model = combined_model(base_model_resnet, base_model_inception)

# %%
# Compile the combined model
optimizer = SGD(learning_rate=0.00096)

combined_model.compile(
    optimizer=optimizer, 
    loss="mean_squared_error",
    metrics=['mean_absolute_error', 'mean_squared_error']
)

# %%
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(
    monitor='val_mean_squared_error',
    patience=3,
    verbose=1, 
    factor=0.2,
    min_lr=0.000001
)

# %%
# Fit the combined model
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
train_features_resnet, train_labels_resnet = next(train_generator)
train_features_inception, _ = next(train_generator)
valid_features_resnet, valid_labels_resnet = next(valid_generator)
valid_features_inception, _ = next(valid_generator)

# Fit the combined model
history_combined = combined_model.fit(
    [train_features_resnet, train_features_inception],  # Pass features for both models
    train_labels_resnet,  # Use labels from any one model
    epochs=30,
    validation_data=([valid_features_resnet, valid_features_inception], valid_labels_resnet),  # Pass features and labels for both models
    verbose=2, 
    callbacks=[learning_rate_reduction, early_stopping],
)

# %% [markdown]
# # 4. Evaluate the model
# ## 4.1 Training and validation curves

# %%
# Plot the loss and accuracy curves for training and validation 
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(history_combined.history['loss'], color='b', label="Training loss")
ax.plot(history_combined.history['val_loss'], color='r', label="Validation loss")
ax.set_ylim(top=np.max(history_combined.history['val_loss'])*1.2, bottom=0)
legend = ax.legend(loc='best', shadow=True)
plt.show()

# Predict on entire validation set using the combined model
valid_generator.reset()
all_labels = []
all_pred = []
for i in range(len(valid_generator)):
    x = next(valid_generator)
    pred_i = combined_model.predict([x[0], x[0]])[:,0]  # Pass the same input twice for both models
    labels_i = x[1]
    all_labels.append(labels_i)
    all_pred.append(pred_i)
    print("Actual Count:", labels_i)
    print("Predicted Count:", pred_i)

cat_labels = np.concatenate(all_labels)
cat_pred = np.concatenate(all_pred)
# predictions_combined = combined_model.predict([valid_features_resnet, valid_features_inception])[:, 0]

# Plot scatter plot with correlation line
df_predictions_combined = pd.DataFrame({'True values': cat_labels, 'Predicted values': cat_pred})
ax = df_predictions_combined.plot.scatter('True values', 'Predicted values', alpha=0.5, s=14, figsize=(9,9))
ax.grid(axis='both')
add_one_to_one_correlation_line(ax)
ax.set_title('Validation')
plt.show()

# Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
mae_combined = np.mean(np.abs(cat_labels - cat_pred))
mse_combined = mean_squared_error(*df_predictions_combined.T.values)
print(f'Mean Absolute Error (MAE) of the combined model: {mae_combined:.2f}')
print(f'Mean Squared Error (MSE) of the combined model: {mse_combined:.2f}')