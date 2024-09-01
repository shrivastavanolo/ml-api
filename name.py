import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib
import numpy as np

# Load datasets
rural_df = pd.read_csv('rural_budget_allocation_dataset.csv')
urban_df = pd.read_csv('urban_budget_allocation_dataset.csv')

# Add a 'Region' column to identify rural and urban households
rural_df['Region'] = 'Rural'
urban_df['Region'] = 'Urban'

# Merge datasets
combined_df = pd.concat([rural_df, urban_df], ignore_index=True)

from sklearn.preprocessing import LabelEncoder, StandardScaler
# Define features and target variables
X = combined_df[['Income']]
y = combined_df[['HousingBudget',
'TransportationBudget',
'FoodBudget',
'UtilitiesBudget',
'EntertainmentBudget','SavingsBudget']]


if __name__=='__main__':
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Define the neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(y_train.shape[1])  # Output layer (number of output features)
    ])

    print(X_train.shape[1],y_train.shape[1])

    # Compile the model
    model.compile(optimizer='Adam', loss='mean_squared_error')

    # Define callbacks for checkpointing
    checkpoint_filepath = 'model_checkpoint.keras'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    # Train the model
    history = model.fit(X_train, y_train,
                        epochs=1000,
                        batch_size=32,
                        validation_data=(X_val, y_val),
                        callbacks=[model_checkpoint_callback])

    # After training, load the best checkpointed model
    model.load_weights(checkpoint_filepath)

    # Predict on the test set
    y_pred = model.predict(X_test)
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
    r2 = r2_score(y_test, y_pred)


    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Root Mean Squared Error (RMSE): {rmse}')
    print(f'R-squared (R2): {r2}')

    # Save the trained model
    model_filename = 'budget_allocation_model_nn.keras'
    model.save(model_filename)

    print(f"Trained Neural Network model saved as {model_filename}")