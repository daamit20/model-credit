import pandas as pd
import xgboost as xgb
import pickle
import os


# Cargar la tabla transformada
def lee_csv(filename):
    df = pd.read_csv(os.path.join("../data/processed", filename))
    # df.drop("Id", axis=1, inplace=True)
    X_train = df.drop(["SalePrice"], axis=1)
    y_train = df[["SalePrice"]]
    print(filename, " cargado correctamente")
    # Entrenamos el modelo con toda la muestra
    xgb_mod = xgb.XGBRegressor(
        colsample_bytree=0.4603,
        gamma=0.0468,
        learning_rate=0.05,
        max_depth=3,
        min_child_weight=1.7817,
        n_estimators=2200,
        reg_alpha=0.4640,
        reg_lambda=0.8571,
        subsample=0.5213,
        silent=1,
        random_state=7,
        nthread=-1,
    )

    # xgb_mod=xgb.XGBClassifier(max_depth=2, n_estimators=50, objective='binary:logistic', seed=0, silent=True, subsample=.8)
    xgb_mod.fit(X_train, y_train)
    print("Modelo entrenado")
    # Guardamos el modelo entrenado para usarlo en produccion
    package = "../models/best_model.pkl"
    pickle.dump(xgb_mod, open(package, "wb"))
    print("Modelo exportado correctamente en la carpeta models")


# Entrenamiento completo
def main():
    lee_csv("procesado_train.csv")
    print("Finalizó el entrenamiento del Modelo")


if __name__ == "__main__":
    main()
