import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "properties.csv"
MODEL_DIR = BASE_DIR / "model"

SALE_MODEL_PATH = MODEL_DIR / "sale_price_model.joblib"
SALE_METADATA_PATH = MODEL_DIR / "sale_price_model_metadata.json"

RENT_MODEL_PATH = MODEL_DIR / "rent_price_model.joblib"
RENT_METADATA_PATH = MODEL_DIR / "rent_price_model_metadata.json"


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def normalize_booleans(df: pd.DataFrame, bool_cols: list[str]) -> pd.DataFrame:
    truthy = {"true", "1", "yes", "y", "t"}
    falsy = {"false", "0", "no", "n", "f"}

    for col in bool_cols:
        if col not in df.columns:
            continue

        if df[col].dtype == bool:
            df[col] = df[col].astype(int)
            continue

        def convert_value(x):
            if pd.isna(x):
                return np.nan
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                return int(x)
            text = str(x).strip().lower()
            if text in truthy:
                return 1
            if text in falsy:
                return 0
            return np.nan

        df[col] = df[col].apply(convert_value)

    return df


def drop_all_nan_numeric_columns(
    df: pd.DataFrame, categorical_features: list[str], numeric_features: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    valid_numeric = []
    for col in numeric_features:
        if col in df.columns and df[col].notna().any():
            valid_numeric.append(col)

    valid_categorical = [col for col in categorical_features if col in df.columns]
    return df, valid_categorical, valid_numeric


def prepare_training_data(
    df: pd.DataFrame,
    listing_mode: str,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str]]:
    if "listing_type" not in df.columns:
        raise ValueError("The dataset must contain a 'listing_type' column.")

    df = df.copy()
    df["listing_type"] = df["listing_type"].astype(str).str.strip().str.lower()
    listing_mode = listing_mode.strip().lower()

    if listing_mode not in {"sale", "rent"}:
        raise ValueError("listing_mode must be either 'sale' or 'rent'.")

    df = df[df["listing_type"].str.contains(listing_mode, na=False)].copy()

    if df.empty:
        raise ValueError(f"No {listing_mode} listings found in the dataset.")

    if "price_usd" not in df.columns:
        raise ValueError("The dataset must contain 'price_usd' as the target column.")

    df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
    if "area_sqm" in df.columns:
        df["area_sqm"] = pd.to_numeric(df["area_sqm"], errors="coerce")

    df = df[df["price_usd"].notna()].copy()
    df = df[df["price_usd"] > 0].copy()

    if "area_sqm" in df.columns:
        df = df[df["area_sqm"].fillna(0) > 0].copy()

    bool_cols = [
        "furnished",
        "balcony",
        "elevator",
        "generator_available",
        "heating",
        "ac",
        "solar_ready",
    ]
    df = normalize_booleans(df, bool_cols)

    categorical_features = [
        "property_type",
        "city",
        "district",
        "neighborhood",
        "state_electricity",
        "generator_cap",
        "city_water",
        "tank_capacity",
        "filtration",
        "traffic_level",
        "quiet_hours",
        "market_feature_source",
        "macro_source_label",
        "status",
    ]

    numeric_features = [
        "area_sqm",
        "bedrooms",
        "bathrooms",
        "floor_number",
        "total_floors",
        "parking_spaces",
        "latitude",
        "longitude",
        "year_built",
        "feature_listing_year",
        "feature_listing_month",
        "feature_listing_quarter",
        "macro_inflation_cpi_pct",
        "macro_real_gdp_growth_pct",
        "macro_debt_to_gdp_pct",
        "macro_current_account_pct_gdp",
        "market_city_listing_count",
        "market_city_avg_ppsqm",
        "market_city_median_ppsqm",
        "market_district_listing_count",
        "market_district_avg_ppsqm",
        "market_district_median_ppsqm",
        "market_city_type_listing_count",
        "market_city_type_avg_ppsqm",
        "market_city_type_median_ppsqm",
        "market_district_type_listing_count",
        "market_district_type_avg_ppsqm",
        "market_district_type_median_ppsqm",
        "market_neighborhood_listing_count",
        "market_neighborhood_avg_ppsqm",
        "market_neighborhood_median_ppsqm",
        "market_local_avg_ppsqm",
        "market_local_listing_count",
        "feature_downtown_distance_km_proxy",
        "feature_traffic_score",
        "feature_quiet_score",
        "furnished",
        "balcony",
        "elevator",
        "generator_available",
        "heating",
        "ac",
        "solar_ready",
    ]

    leakage_or_unused = {
        "id",
        "title",
        "listing_type",
        "address",
        "water_supply",
        "downtown_distance",
        "market_position",
        "pricing_reason",
        "price_history",
        "description",
        "fair_price_min",
        "fair_price_max",
        "price_per_sqm",
        "agent_id",
        "solar_ready_text",
        "created_at",
    }

    requested = categorical_features + numeric_features
    available_features = [
        col for col in requested if col in df.columns and col not in leakage_or_unused
    ]

    if not available_features:
        raise ValueError("No valid features were found in the dataset.")

    X = df[available_features].copy()
    y = df["price_usd"].copy()

    X, available_categorical, available_numeric = drop_all_nan_numeric_columns(
        X, categorical_features, numeric_features
    )

    final_features = available_categorical + available_numeric
    X = X[final_features].copy()

    return X, y, available_categorical, available_numeric, final_features


def build_preprocessor(
    categorical_features: list[str], numeric_features: list[str]
) -> ColumnTransformer:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    return {
        "model_name": name,
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(root_mean_squared_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "pipeline": pipeline,
    }


def train_for_mode(df: pd.DataFrame, listing_mode: str, model_path: Path, metadata_path: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"Training model for: {listing_mode.upper()}")
    print(f"{'=' * 70}")

    X, y, categorical_features, numeric_features, used_features = prepare_training_data(
        df, listing_mode=listing_mode
    )

    print(f"Rows used for training: {len(X)}")
    print(f"Number of features used: {len(used_features)}")
    print("Categorical features:")
    print(categorical_features)
    print("Numeric features:")
    print(numeric_features)

    if len(X) < 10:
        raise ValueError(
            f"Not enough {listing_mode} rows to train a reliable model. Found only {len(X)} rows."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    preprocessor = build_preprocessor(categorical_features, numeric_features)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            min_samples_leaf=10,
            l2_regularization=0.0,
            random_state=42,
        ),
    }

    results = []

    print("\nTraining models...\n")
    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

        metrics = evaluate_model(
            name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
        results.append(metrics)

        print(f"{model_name}")
        print(f"  MAE :  {metrics['mae']:,.2f}")
        print(f"  RMSE:  {metrics['rmse']:,.2f}")
        print(f"  R²  :  {metrics['r2']:.4f}")
        print()

    results_sorted = sorted(results, key=lambda d: (d["rmse"], d["mae"]))
    best = results_sorted[0]

    print("Best model selected:")
    print(f"  {best['model_name']}")
    print(f"  MAE :  {best['mae']:,.2f}")
    print(f"  RMSE:  {best['rmse']:,.2f}")
    print(f"  R²  :  {best['r2']:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipeline"], model_path)

    metadata = {
        "listing_mode": listing_mode,
        "model_name": best["model_name"],
        "target": "price_usd",
        "n_rows_used": int(len(X)),
        "n_train_rows": int(len(X_train)),
        "n_test_rows": int(len(X_test)),
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "used_features": used_features,
        "metrics": {
            "mae": best["mae"],
            "rmse": best["rmse"],
            "r2": best["r2"],
        },
        "data_path": str(DATA_PATH),
        "model_path": str(model_path),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved model to:")
    print(model_path)
    print("Saved metadata to:")
    print(metadata_path)

    example_input = X_test.iloc[[0]].copy()
    example_prediction = best["pipeline"].predict(example_input)[0]

    print("\nExample prediction on one held-out test row:")
    print(example_input.to_dict(orient="records")[0])
    print(f"Predicted price_usd: ${example_prediction:,.2f}")
    print(f"Actual price_usd   : ${float(y_test.iloc[0]):,.2f}")


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    train_for_mode(df, "sale", SALE_MODEL_PATH, SALE_METADATA_PATH)
    train_for_mode(df, "rent", RENT_MODEL_PATH, RENT_METADATA_PATH)


if __name__ == "__main__":
    main()