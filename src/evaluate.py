import pandas as pd
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import *
import os
import numpy as np


def eval_model(filename):
    df = pd.read_csv(os.path.join("../data/processed", filename))
    print(filename, " cargado correctamente")
    package = "../models/best_model.pkl"
    model = pickle.load(open(package, "rb"))
    print("Modelo importado correctamente")
    # Predecimos sobre el set de datos de validación
    X = df.drop(["SalePrice"], axis=1)
    y = df[["SalePrice"]]

    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")
    print(f"R^2 Score: {r2:.2f}")


# Validación desde el inicio
def main():
    df = eval_model("procesado_validation.csv")
    print("Finalizó la validación del Modelo")


if __name__ == "__main__":
    main()
