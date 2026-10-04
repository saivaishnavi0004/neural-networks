import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ==================================================
#                LOAD DATASET
# ==================================================

data = pd.read_csv("student_performance_dataset.csv")

print("\n================ DATASET ================")
print(data.head())

print("\nTotal Students:", len(data))


# ==================================================
#                INPUT FEATURES
# ==================================================

X = data[
    [
        "Attendance_%",
        "Study_Hours_per_Week",
        "Assignment_Marks(/20)",
        "Internal_Marks(/30)",
        "Previous_Marks(/100)"
    ]
]

y = data["Final_Marks(/100)"]


# ==================================================
#                TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n================ DATA SPLIT ================")
print("Training Students:", len(X_train))
print("Testing Students :", len(X_test))


# Get names of test students
test_names = data.loc[X_test.index, "Student_Name"]


# ==================================================
#                STANDARDIZATION
# ==================================================

X_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)

X_test_scaled = X_scaler.transform(X_test)


# Standardize target
y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
).flatten()


# ==================================================
#                BUILD ANN MODEL
# ==================================================

model = Sequential([
    Input(shape=(5,)),
    Dense(32, activation="relu"),
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1)
])


# ==================================================
#                COMPILE MODEL
# ==================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


# ==================================================
#                EARLY STOPPING
# ==================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=30,
    restore_best_weights=True
)


# ==================================================
#                TRAIN MODEL
# ==================================================

print("\n================ MODEL TRAINING ================")

history = model.fit(
    X_train_scaled,
    y_train_scaled,
    epochs=500,
    batch_size=8,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)


# ==================================================
#                TEST PREDICTION
# ==================================================

y_pred_scaled = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# Convert predictions back to original marks
y_pred = y_scaler.inverse_transform(
    y_pred_scaled.reshape(-1, 1)
).flatten()


# Keep predicted marks between 0 and 100
y_pred = np.clip(y_pred, 0, 100)


# ==================================================
#                MODEL EVALUATION
# ==================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)


print("\n==========================================")
print("          MODEL PERFORMANCE")
print("==========================================")

print("MAE      :", round(mae, 2))
print("MSE      :", round(mse, 2))
print("RMSE     :", round(rmse, 2))
print("R2 Score :", round(r2, 2))


# ==================================================
#                ACTUAL VS PREDICTED
# ==================================================

result = pd.DataFrame({
    "Student Name": test_names.values,
    "Actual Final Marks": y_test.values,
    "Predicted Final Marks": np.round(y_pred, 2)
})


print("\n==========================================")
print("        ACTUAL VS PREDICTED")
print("==========================================")

print(result.to_string(index=False))


# ==================================================
#            NEW STUDENT USER INPUT
# ==================================================

print("\n==========================================")
print("        NEW STUDENT PREDICTION")
print("==========================================")

attendance = float(
    input("Enter Attendance (%): ")
)

study_hours = float(
    input("Enter Study Hours per Week: ")
)

assignment_marks = float(
    input("Enter Assignment Marks (/20): ")
)

internal_marks = float(
    input("Enter Internal Marks (/30): ")
)

previous_marks = float(
    input("Enter Previous Marks (/100): ")
)


# ==================================================
#            CREATE NEW STUDENT DATA
# ==================================================

new_student = pd.DataFrame({
    "Attendance_%": [attendance],
    "Study_Hours_per_Week": [study_hours],
    "Assignment_Marks(/20)": [assignment_marks],
    "Internal_Marks(/30)": [internal_marks],
    "Previous_Marks(/100)": [previous_marks]
})


# ==================================================
#            SCALE NEW STUDENT DATA
# ==================================================

new_student_scaled = X_scaler.transform(
    new_student
)


# ==================================================
#            PREDICT NEW STUDENT
# ==================================================

new_prediction_scaled = model.predict(
    new_student_scaled,
    verbose=0
).flatten()


# Convert prediction back to original marks
new_prediction = y_scaler.inverse_transform(
    new_prediction_scaled.reshape(-1, 1)
).flatten()[0]


# Keep marks between 0 and 100
new_prediction = np.clip(
    new_prediction,
    0,
    100
)


# ==================================================
#            NEW STUDENT RESULT
# ==================================================

print("\n==========================================")
print("       STUDENT PREDICTION RESULT")
print("==========================================")

print("Attendance           :", attendance, "%")

print("Study Hours/Week     :", study_hours)

print("Assignment Marks     :", assignment_marks, "/20")

print("Internal Marks       :", internal_marks, "/30")

print("Previous Marks       :", previous_marks, "/100")

print("------------------------------------------")

print(
    "Predicted Final Marks:",
    round(new_prediction, 2),
    "/100"
)

print("==========================================")


# ==================================================
#                GRAPH 1
#          ACTUAL VS PREDICTED
# ==================================================

plt.figure(figsize=(12, 6))

plt.plot(
    test_names.values,
    y_test.values,
    marker="o",
    label="Actual Final Marks"
)

plt.plot(
    test_names.values,
    y_pred,
    marker="o",
    label="Predicted Final Marks"
)

plt.xlabel("Student Name")

plt.ylabel("Final Marks")

plt.title("Actual vs Predicted Final Marks")

plt.xticks(
    rotation=45
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()


# ==================================================
#                GRAPH 2
#       TRAINING VS VALIDATION LOSS
# ==================================================

plt.figure(figsize=(8, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()