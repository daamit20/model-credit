import pandas as pd
import numpy as np
import os


def lee_csv(filename):
    # df = pd.read_csv(os.path.join("../data/raw/", filename)).set_index("Id")
    df = pd.read_csv(os.path.join("../data/raw/", filename))
    print(f"Cargado Correctamente el archivo {filename}")
    return df


def prepara_data(df):
    df_id = df["Id"]
    df.drop("Id", axis=1, inplace=True)
    if "SalePrice" in df.columns:
        df = df.drop(df[(df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)].index)
    df["PoolQC"] = df["PoolQC"].fillna("None")
    df["MiscFeature"] = df["MiscFeature"].fillna("None")
    df["Alley"] = df["Alley"].fillna("None")
    df["Fence"] = df["Fence"].fillna("None")
    df["FireplaceQu"] = df["FireplaceQu"].fillna("None")

    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    for col in ("GarageType", "GarageFinish", "GarageQual", "GarageCond"):
        df[col] = df[col].fillna("None")
    for col in ("GarageYrBlt", "GarageArea", "GarageCars"):
        df[col] = df[col].fillna(0)
    for col in (
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
    ):
        df[col] = df[col].fillna(0)
    for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
        df[col] = df[col].fillna("None")
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df["MSZoning"] = df["MSZoning"].fillna(df["MSZoning"].mode()[0])
    df = df.drop(["Utilities"], axis=1)
    df["Functional"] = df["Functional"].fillna("Typ")
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])
    df["KitchenQual"] = df["KitchenQual"].fillna(df["KitchenQual"].mode()[0])
    df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
    df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
    df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])
    df["MSSubClass"] = df["MSSubClass"].fillna("None")
    # df_na = (df.isnull().sum() / len(df)) * 100
    # df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
    df["MSSubClass"] = df["MSSubClass"].apply(str)
    df["OverallCond"] = df["OverallCond"].astype(str)
    df["YrSold"] = df["YrSold"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)

    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df_filtro = df[df.select_dtypes(include=["int64", "float64"]).columns]
    # df_filtro = df_filtro.dropna(subset=["LotFrontage"])

    return df_filtro


def exporta_data(df, features, filename):
    dfp = df[features]
    dfp.to_csv(os.path.join("../data/processed/", filename))
    print(filename, "exportado correctamente en la carpeta processed")


def main():

    df = lee_csv("train.csv")
    dp_df = prepara_data(df)
    exporta_data(
        dp_df,
        [
            "LotFrontage",
            "LotArea",
            "OverallQual",
            "YearBuilt",
            "YearRemodAdd",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "LowQualFinSF",
            "GrLivArea",
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "PoolArea",
            "MiscVal",
            "TotalSF",
            "SalePrice",
        ],
        "procesado_train.csv",
    )
    df = lee_csv("validation.csv")
    dp_df = prepara_data(df)
    exporta_data(
        dp_df,
        [
            "LotFrontage",
            "LotArea",
            "OverallQual",
            "YearBuilt",
            "YearRemodAdd",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "LowQualFinSF",
            "GrLivArea",
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "PoolArea",
            "MiscVal",
            "TotalSF",
            "SalePrice",
        ],
        "procesado_validation.csv",
    )
    df = lee_csv("test.csv")
    dp_df = prepara_data(df)
    exporta_data(
        dp_df,
        [
            "LotFrontage",
            "LotArea",
            "OverallQual",
            "YearBuilt",
            "YearRemodAdd",
            "MasVnrArea",
            "BsmtFinSF1",
            "BsmtFinSF2",
            "BsmtUnfSF",
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "LowQualFinSF",
            "GrLivArea",
            "BsmtFullBath",
            "BsmtHalfBath",
            "FullBath",
            "HalfBath",
            "BedroomAbvGr",
            "KitchenAbvGr",
            "TotRmsAbvGrd",
            "Fireplaces",
            "GarageYrBlt",
            "GarageCars",
            "GarageArea",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "PoolArea",
            "MiscVal",
            "TotalSF",
        ],
        "procesado_test.csv",
    )


if __name__ == "__main__":
    main()
