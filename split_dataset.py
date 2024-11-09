import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Define file paths
features_path = r'C:\Users\mohse\cdo-idp-dev\data\My_cnn\train.csv'
y_values_path = r'C:\Users\mohse\cdo-idp-dev\data\My_cnn\train_result.csv'
output_dir = r'C:\Users\mohse\cdo-idp-dev\data\My_cnn\split_data'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the datasets
features = pd.read_csv(features_path)
y_values = pd.read_csv(y_values_path)

# Split the data into train, validation, and test sets (e.g., 70% train, 15% validation, 15% test)
X_train, X_temp, y_train, y_temp = train_test_split(features, y_values, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Save the splits into CSV files
X_train.to_csv(os.path.join(output_dir, 'train_features.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'train_labels.csv'), index=False)

X_val.to_csv(os.path.join(output_dir, 'val_features.csv'), index=False)
y_val.to_csv(os.path.join(output_dir, 'val_labels.csv'), index=False)

X_test.to_csv(os.path.join(output_dir, 'test_features.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'test_labels.csv'), index=False)

print("Data has been split and saved successfully!")
