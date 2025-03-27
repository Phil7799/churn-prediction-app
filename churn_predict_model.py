import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
from sklearn.utils import resample

# Optional: Try to import LightGBM
try:
    from lightgbm import LGBMClassifier
    lightgbm_available = True
except ImportError:
    print("LightGBM not installed. Proceeding with Random Forest only.")
    print("To install LightGBM, run: pip install lightgbm")
    lightgbm_available = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load both datasets
churned_data = pd.read_excel(r'C:\Users\philip.otieno\Desktop\ML - AI - Projects\Client - Churn - Prediction Model\churned_clients.xlsx')  # Churned clients
non_churned_data = pd.read_excel(r'C:\Users\philip.otieno\Desktop\ML - AI - Projects\Client - Churn - Prediction Model\active_clients.xlsx')  # Non-churned clients

# Label the datasets
churned_data['churned'] = 1
non_churned_data['churned'] = 0

# Apply feature engineering to both datasets individually
for df in [churned_data, non_churned_data]:
    df['last_seen_dated'] = pd.to_datetime(df['last_seen_dated'])
    df['registered_on'] = pd.to_datetime(df['registered_on'])
    df['tenure_days'] = (df['last_seen_dated'] - df['registered_on']).dt.days
    df['has_trips'] = (df['Trips_'] > 0).astype(int)
    df['Cancellations_per_Trip'] = (df['DriverCancellations_'] + df['RiderCancellations_']) / df['Trips_'].replace(0, np.nan)
    df['Cancellations_per_Trip'] = df['Cancellations_per_Trip'].fillna(0)
    df['Timeouts_per_Request'] = df['Timeouts_'] / df['Total_requests'].replace(0, np.nan)
    df['Timeouts_per_Request'] = df['Timeouts_per_Request'].fillna(0)
    df['NoDrivers_per_Request'] = df['no_drivers_found'] / df['Total_requests'].replace(0, np.nan)
    df['NoDrivers_per_Request'] = df['NoDrivers_per_Request'].fillna(0)
    df['Trip_Frequency'] = df['Total_requests'] / (df['tenure_days'].replace(0, np.nan) / 30)
    df['Trip_Frequency'] = df['Trip_Frequency'].fillna(0)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Concatenate the datasets
data = pd.concat([churned_data, non_churned_data], axis=0, ignore_index=True)

# Initial feature set
initial_features = [
    'no_drivers_found', 'Timeouts_', 'FulfillmentRate', 'DriverCancellations_',
    'has_trips', 'Timeouts_per_Request', 'tenure_days', 'Trip_Frequency'
]
X = data[initial_features].copy()
y = data['churned']

# Check for missing values and ensure no NaNs or infs
print("\nMissing values in features:")
print(X.isnull().sum())
print("\nMissing values in targets:")
print(y.isnull().sum())

# Ensure no infinite values
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.fillna(X.median())

# Check correlation to remove multicollinearity
correlation_matrix = X.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Remove highly correlated features (threshold > 0.7)
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.7)]
X_reduced = X.drop(columns=to_drop) if to_drop else X
print("\nFeatures after removing highly correlated ones:", X_reduced.columns.tolist())

# Save the list of dropped columns
joblib.dump(to_drop, 'dropped_columns.pkl')

# Add polynomial and interaction features (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_reduced)
feature_names = poly.get_feature_names_out(X_reduced.columns)

# Save the PolynomialFeatures object
joblib.dump(poly, 'poly_features.pkl')

# Feature selection using SelectKBest (k=8)
selector = SelectKBest(score_func=f_classif, k=8)
X_selected = selector.fit_transform(X_poly, y)
selected_mask = selector.get_support()
selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
print("\nSelected Features:", selected_features)

# Save the SelectKBest object
joblib.dump(selector, 'selector.pkl')

# Impute and scale the selected data
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_selected)

# Save the SimpleImputer object
joblib.dump(imputer, 'imputer.pkl')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Save the StandardScaler object
joblib.dump(scaler, 'scaler.pkl')

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Sample the training data to speed up hyperparameter tuning (50% sample)
X_train_sample, y_train_sample = resample(X_train, y_train, n_samples=int(0.5 * len(X_train)), stratify=y_train, random_state=42)

# Random Forest with RandomizedSearchCV for faster tuning
param_dist_rf = {
    'n_estimators': [200, 300, 400],
    'max_depth': [20, 30, 40],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [1, 2],
    'class_weight': ['balanced', {0: 1, 1: 2}]
}
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
random_search_rf = RandomizedSearchCV(rf, param_distributions=param_dist_rf, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
random_search_rf.fit(X_train_sample, y_train_sample)
print("Best parameters for Random Forest:", random_search_rf.best_params_)

# Retrain the best model on the full training set
best_rf = RandomForestClassifier(**random_search_rf.best_params_, random_state=42, n_jobs=-1)
best_rf.fit(X_train, y_train)

# Save the trained Random Forest model
joblib.dump(best_rf, 'churn_model_rf.pkl')

# Cross-validation scores for Random Forest (on full data)
cv_scores = cross_val_score(best_rf, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
print("\nRandom Forest Cross-Validation Accuracy Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean(), "±", cv_scores.std())

# Evaluate the model
y_pred_rf = best_rf.predict(X_test)
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]
print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Optional LightGBM fallback (if installed and accuracy < 92%)
if lightgbm_available and best_rf.score(X_test, y_test) < 0.92:
    print("\nRandom Forest accuracy below 92%. Switching to LightGBM...")
    param_dist_lgbm = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 63],
        'scale_pos_weight': [1, 2]
    }
    lgbm = LGBMClassifier(random_state=42, n_jobs=-1)
    random_search_lgbm = RandomizedSearchCV(lgbm, param_distributions=param_dist_lgbm, n_iter=10, cv=3, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search_lgbm.fit(X_train_sample, y_train_sample)
    print("Best parameters for LightGBM:", random_search_lgbm.best_params_)
    
    # Retrain on full data
    best_lgbm = LGBMClassifier(**random_search_lgbm.best_params_, random_state=42, n_jobs=-1)
    best_lgbm.fit(X_train, y_train)
    
    # Save the trained LightGBM model
    joblib.dump(best_lgbm, 'churn_model_lgbm.pkl')
    
    # Cross-validation scores for LightGBM
    cv_scores_lgbm = cross_val_score(best_lgbm, X_train, y_train, cv=3, scoring='accuracy', n_jobs=-1)
    print("\nLightGBM Cross-Validation Accuracy Scores:", cv_scores_lgbm)
    print("Mean CV Accuracy:", cv_scores_lgbm.mean(), "±", cv_scores_lgbm.std())
    
    # Evaluate the model
    y_pred_lgbm = best_lgbm.predict(X_test)
    y_pred_proba_lgbm = best_lgbm.predict_proba(X_test)[:, 1]
    print("\nClassification Report for LightGBM:")
    print(classification_report(y_test, y_pred_lgbm))