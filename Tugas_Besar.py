# import library dan module
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor # Added this import
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import OneHotEncoder


# Import dataset
path = "https://drive.google.com/uc?export=download&id=1WxD4Mld0Rbgj_FSpzQ0An8dzKAIp7sEs"
df = pd.read_csv(path, encoding='latin1')

# Fix column name
df.columns = df.columns.str.lower()
df.columns = df.columns.str.replace(' ', '_')

# Remove next line
for col in df.select_dtypes(include='object'):
    df[col] = df[col].str.strip()


# Remove duplicated youtuber
df = df.drop_duplicates(subset=['youtuber'], keep='first')

# Remove negatif value in subscribers, video views and uploads
mask_valid = (df["subscribers"] >= 0) & (df["video_views"] >= 0) & (df["uploads"] >= 0)
invalid_rows = df.shape[0] - df[mask_valid].shape[0]
print(f"\nJumlah baris dengan nilai negatif (akan dibuang): {invalid_rows}")

df = df[mask_valid].copy()

cat_fill_unknown = ["category", "channel_type", "Country", "Abbreviation"]
for col in cat_fill_unknown:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

df.dropna(subset=['created_year'], inplace=True)
df.dropna(subset=['created_month'], inplace=True)
df.dropna(subset=['created_date'], inplace=True)

current_year = 2025  # misal asumsi tahun sekarang
df["channel_age_years"] = current_year - df["created_year"]

target_col = "highest_yearly_earnings"

print("Shape data untuk ML (setelah drop NaN fitur & target):", df.shape)

X = df.drop(columns=[target_col])
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42
)

print(f"Jumlah data train: {X_train.shape[0]}")
print(f"Jumlah data test : {X_test.shape[0]}")

num_cols = X_train.select_dtypes(include=['float64','int64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

preprocess = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ('preprocess', preprocess),
    ('select', SelectKBest(score_func=f_regression, k=10)),
    ('model', model)
])


# CV evaluation
scores = cross_val_score(
    pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1
)

print("CV R² mean:", scores.mean())
print("CV R² std:", scores.std())

# # ======== FIT MODEL DULU ========
pipeline.fit(X_train, y_train)

# ======== PREDIKSI ========
y_train_pred = pipeline.predict(X_train)
y_test_pred  = pipeline.predict(X_test)

# ======== SCORE R2 ========
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# ======== ERROR METRICS ========
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)

# ======== CETAK HASIL ========
print("=== Performance Test ===")
print(f"Train R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")
print("-------------------------")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")

# ======== SAMPLE OUTPUT ========
print("\n=== Sample Predictions ===")
for i in range(5):
    print(f"Actual: {y_test.iloc[i]:.2f}   Predicted: {y_test_pred[i]:.2f}")

