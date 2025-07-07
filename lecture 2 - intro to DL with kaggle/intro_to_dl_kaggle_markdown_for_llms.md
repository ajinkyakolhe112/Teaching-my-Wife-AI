# Converted from intro_to_dl_kaggle.ipynb - Markdown format optimized for LLM readability

```python
# SINGLE NEURON

from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

```python
# SIMPLE NEURAL NETWORK OF 3 Layers
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
```

```python
# SIMPLE PROBLEM - Housing Price Prediction Regression
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

# Load the California Housing dataset
(X_train, y_train), (X_valid, y_valid) = keras.datasets.california_housing.load_data(
    version="large", path="california_housing.npz", test_split=0.2, seed=113
)

print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation data shape:", X_valid.shape)
print("Validation labels shape:", y_valid.shape)

# Create the model with correct input shape (8 features)
model = keras.Sequential([
    layers.Dense(512, activation='relu', input_shape=[8]),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1),
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='mae',
)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=10,
)

# Plot the training history
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))
```