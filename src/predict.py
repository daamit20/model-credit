import pandas as pd
import xgboost as xgb
import pickle
import os


# Cargar la tabla transformada
def score_model(filename, test, file_final):

    X_val = pd.read_csv(os.path.join("../data/processed", filename))
    print(filename, " cargado correctamente")
    df_origen = pd.read_csv(os.path.join("../data/raw", test))
    print(df_origen.shape)
    package = "../models/best_model.pkl"
    model = pickle.load(open(package, "rb"))
    print("Modelo importado correctamente")

    predictions = model.predict(X_val)
    print("A:")

    pred_df = pd.DataFrame(predictions, columns=["Prediction"])

    combinado = pd.concat([df_origen, pred_df], axis=1)

    # Exportar las predicciones a un archivo CSV
    combinado.to_csv(os.path.join("../data/scores/", file_final))
    print(f"{file_final} exportado correctamente en la carpeta scores")


# Scoring desde el inicio
def main():
    df = score_model("procesado_test.csv", "test.csv", "Final_Prediccion.csv")
    print("Finalizó el Scoring del Modelo")


if __name__ == "__main__":
    main()
