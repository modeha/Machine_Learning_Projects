# Import necessary libraries
import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.utils import plot_model
import pydot  # Import pydot explicitly

# Force the script to use `pydot` for both `pydot` and `pydotplus` modules
sys.modules['pydot'] = pydot
sys.modules['pydotplus'] = pydot

# Define model parameters
lstm_units = 50
dropout_rate = 0.2
input_shape = (10, 5)  # Example input shape (timesteps, features)

# Define the Sequential model
model = Sequential([
    LSTM(lstm_units, input_shape=input_shape),
    Dropout(dropout_rate),
    Dense(1, activation='relu')
])

# Plot and save the model architecture to a file
output_file = 'lstm_model.png'  # The file where the diagram will be saved
plot_model(model, to_file=output_file, show_shapes=True, show_layer_names=True)

# Check if the file was created successfully
if os.path.exists(output_file):
    print(f"Model architecture diagram saved successfully as {output_file}")
else:
    print("Failed to save the model architecture diagram.")
