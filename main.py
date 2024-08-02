#pip install pandas numpy scikit-learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

# Read data from CSV file
data = pd.read_csv('road_repair_costs_converted_corrected.csv')  # Ensure the file is in the correct location

# Convert road type from text to numbers
label_encoder = LabelEncoder()
data['road_type'] = label_encoder.fit_transform(data['road_type'])

# Separate Features and Target
X = data[['road_type', 'width', 'length', 'depth']]
y = data['cost']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Decision Tree model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Check the model score
print("Training Score:", model.score(X_train, y_train))

def predict_repair_cost(road_type, width_cm, length_cm, depth_cm):
    # Convert road type from text to numbers
    road_type_encoded = label_encoder.transform([road_type])[0]
    # Convert cm to meters
    width_m = width_cm / 100
    length_m = length_cm / 100
    depth_m = depth_cm / 100
    # Create numpy array from input
    input_data = np.array([[road_type_encoded, width_m, length_m, depth_m]])
    # Predict cost
    predicted_cost = model.predict(input_data)
    return predicted_cost[0]

def main():
    road_type = input("Please select the road type (Concrete, Asphalt, Gravel): ")
    width_cm = float(input("Please enter the width of the hole (centimeters): "))
    length_cm = float(input("Please enter the length of the hole (centimeters): "))
    depth_cm = float(input("Please enter the depth of the hole (centimeters): "))

    cost = predict_repair_cost(road_type, width_cm, length_cm, depth_cm)
    print(f"The cost of repairing a {road_type} road for a hole with a volume of {width_cm*length_cm*depth_cm/1000000:.2f} cubic meters is {cost:.2f} baht")

if __name__ == "__main__":
    main()
