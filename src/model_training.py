import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv("data/insurance.csv")

# Encode categorical columns
le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop("charges", axis=1)
y = df["charges"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning
params = {
    "n_estimators": [50,100,150],
    "learning_rate": [0.01,0.1,1]
}

grid = GridSearchCV(
    AdaBoostRegressor(random_state=42),
    params,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

model = grid.best_estimator_

# Evaluation
y_pred = model.predict(X_test_scaled)

print("MSE :", mean_squared_error(y_test, y_pred))
print("R2 Score :", r2_score(y_test, y_pred))
print("Best Params :", grid.best_params_)

# Save
pickle.dump(
    model,
    open("models/adaboost_regressor.pkl", "wb")
)

pickle.dump(
    scaler,
    open("models/scaler.pkl", "wb")
)

print("Model Saved Successfully")