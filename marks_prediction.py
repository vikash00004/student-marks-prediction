# Student Marks Prediction System
# Machine Learning Project using Linear Regression

from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
# Hours studied -> Marks obtained
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
marks = np.array([35, 40, 45, 50, 55, 60, 65, 70, 80, 90])

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(hours, marks)

print("====================================")
print("   STUDENT MARKS PREDICTION SYSTEM")
print("====================================")

# Take input from user
try:
    study_hours = float(input("Enter number of study hours: "))

    if study_hours < 0:
        print("Study hours cannot be negative.")
    else:
        # Predict marks
        predicted_marks = model.predict([[study_hours]])[0]

        # Keep prediction within 0-100
        predicted_marks = max(0, min(100, predicted_marks))

        print(f"Predicted Marks: {predicted_marks:.2f}")

except ValueError:
    print("Please enter a valid number.")
