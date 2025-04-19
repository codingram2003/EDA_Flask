import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Create a simple random forest model
X = np.array([[60], [70], [80], [90]])  # Example gain values
y = np.array([
    # W1, L1, W3, L3, W5, L5, W6, L6, W7, L7, IB, CC
    [20e-6, 0.3e-6, 15e-6, 0.3e-6, 18e-6, 0.4e-6, 120e-6, 0.2e-6, 100e-6, 0.2e-6, 10e-6, 5e-12],
    [22e-6, 0.33e-6, 17e-6, 0.33e-6, 19e-6, 0.45e-6, 126e-6, 0.23e-6, 104e-6, 0.23e-6, 12e-6, 6e-12],
    [24e-6, 0.35e-6, 19e-6, 0.35e-6, 20e-6, 0.5e-6, 130e-6, 0.25e-6, 110e-6, 0.25e-6, 14e-6, 7e-12],
    [26e-6, 0.37e-6, 21e-6, 0.37e-6, 22e-6, 0.55e-6, 135e-6, 0.27e-6, 115e-6, 0.27e-6, 16e-6, 8e-12],
])

# Train a simple model
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Test model created and saved as 'random_forest_model.pkl'")
