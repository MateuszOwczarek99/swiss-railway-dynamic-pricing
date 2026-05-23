"""
data cleaning and preprocessing pipeline
we prepare raw data for training our demand prediction model
this follows standard ml engineering practices
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ============================================================================
# STEP 1: LOAD THE REAL DATASET
# ============================================================================

def step1_load_data(filepath='data/raw/railway_prices.csv'):
    """load the real dataset from raw data folder"""
    print("=" * 70)
    print("STEP 1: LOADING REAL DATASET")
    print("=" * 70)
    
    df = pd.read_csv(filepath)
    print(f"dataset loaded successfully")
    print(f"shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"columns: {list(df.columns)}")
    
    # convert timestamp to datetime if not already
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


# ============================================================================
# STEP 2: SANITY CHECK OF THE DATA
# ============================================================================

def step2_sanity_check(df):
    """perform sanity checks to understand data quality"""
    print("\n" + "=" * 70)
    print("STEP 2: SANITY CHECK OF THE DATA")
    print("=" * 70)
    
    # 2.1 check data types
    print("\n[2.1] data types:")
    print(df.dtypes)
    
    # 2.2 basic statistics
    print("\n[2.2] basic statistics:")
    print(df.describe())
    
    # 2.3 check for negative values in price
    print("\n[2.3] checking for negative values:")
    negative_prices = (df['price'] < 0).sum()
    print(f"negative prices found: {negative_prices}")
    
    # 2.4 check demand range
    print("\n[2.4] checking demand range:")
    demand_out_of_range = ((df['demand'] < 0) | (df['demand'] > 1)).sum()
    print(f"demand values outside [0,1]: {demand_out_of_range}")
    
    # 2.5 check days to departure
    print("\n[2.5] checking days to departure:")
    print(f"min days: {df['days_to_departure'].min()}")
    print(f"max days: {df['days_to_departure'].max()}")
    
    return df


# ============================================================================
# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================

def step3_exploratory_data_analysis(df):
    """perform eda to understand patterns and relationships"""
    print("\n" + "=" * 70)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    # create eda directory for saving plots
    os.makedirs('reports/figures', exist_ok=True)
    
    # 3.1 demand distribution analysis
    print("\n[3.1] demand distribution analysis:")
    print(f"mean demand: {df['demand'].mean():.4f}")
    print(f"median demand: {df['demand'].median():.4f}")
    print(f"std demand: {df['demand'].std():.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.hist(df['demand'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('demand')
    plt.ylabel('frequency')
    plt.title('distribution of ticket demand')
    plt.savefig('reports/figures/demand_distribution.png')
    plt.close()
    print("   saved demand distribution plot to reports/figures/")
    
    # 3.2 price analysis by weekend vs weekday
    print("\n[3.2] price analysis by weekend/weekday:")
    weekend_prices = df[df['is_weekend'] == 1]['price'].mean()
    weekday_prices = df[df['is_weekend'] == 0]['price'].mean()
    print(f"   average weekend price: {weekend_prices:.2f} chf")
    print(f"   average weekday price: {weekday_prices:.2f} chf")
    print(f"   weekend premium: {(weekend_prices - weekday_prices):.2f} chf")
    
    # 3.3 correlation analysis
    print("\n[3.3] correlation analysis:")
    numeric_cols = ['price', 'competitor_price', 'demand', 'days_to_departure', 'month']
    correlation = df[numeric_cols].corr()
    print("\n   correlation matrix:")
    print(correlation)
    
    # save correlation heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('feature correlation matrix')
    plt.tight_layout()
    plt.savefig('reports/figures/correlation_heatmap.png')
    plt.close()
    print("   saved correlation heatmap to reports/figures/")
    
    # 3.4 demand vs price relationship
    print("\n[3.4] demand-price relationship:")
    price_demand_corr = df['price'].corr(df['demand'])
    print(f"   correlation between price and demand: {price_demand_corr:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df['price'], df['demand'], alpha=0.5)
    plt.xlabel('price (chf)')
    plt.ylabel('demand')
    plt.title('demand vs price')
    plt.savefig('reports/figures/demand_vs_price.png')
    plt.close()
    print("   saved demand vs price plot to reports/figures/")
    
    return df


# ============================================================================
# STEP 4: MISSING VALUE TREATMENT
# ============================================================================

def step4_handle_missing_values(df):
    """handle missing values using appropriate methods"""
    print("\n" + "=" * 70)
    print("STEP 4: MISSING VALUE TREATMENT")
    print("=" * 70)
    
    # 4.1 check missing values before treatment
    print("\n[4.1] missing values before treatment:")
    missing_before = df.isnull().sum()
    missing_cols = missing_before[missing_before > 0]
    if len(missing_cols) > 0:
        print(missing_cols)
    else:
        print("   no missing values found")
    
    # 4.2 separate numerical and categorical columns
    numerical_cols = ['price', 'competitor_price', 'demand', 'days_to_departure', 'month']
    categorical_cols = ['origin', 'destination']
    
    # 4.3 median imputation for numerical columns
    print("\n[4.2] imputing numerical columns with median:")
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"   imputed {col} with median: {median_value:.2f}")
    
    # 4.4 knn imputation for competitor prices
    if 'competitor_price' in df.columns and df['competitor_price'].isnull().sum() > 0:
        print("\n[4.3] using knn imputation for competitor_price:")
        
        knn_features = ['price', 'days_to_departure', 'is_weekend', 'month']
        knn_data = df[knn_features].copy()
        
        imputer = KNNImputer(n_neighbors=5)
        imputed_values = imputer.fit_transform(knn_data)
        
        missing_mask = df['competitor_price'].isnull()
        df.loc[missing_mask, 'competitor_price'] = imputed_values[missing_mask, 0] * 0.9
        
        print(f"   imputed {missing_mask.sum()} missing competitor prices using knn")
    
    # 4.5 mode imputation for categorical columns
    print("\n[4.4] imputing categorical columns with mode:")
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"   imputed {col} with mode: {mode_value}")
    
    # 4.6 verify no missing values remain
    print("\n[4.5] missing values after treatment:")
    missing_after = df.isnull().sum().sum()
    print(f"   total missing values remaining: {missing_after}")
    
    return df


# ============================================================================
# STEP 5: OUTLIERS TREATMENT
# ============================================================================

def step5_handle_outliers(df):
    """detect and treat outliers using iqr method"""
    print("\n" + "=" * 70)
    print("STEP 5: OUTLIERS TREATMENT")
    print("=" * 70)
    
    original_rows = len(df)
    
    # 5.1 detect price outliers using iqr
    print("\n[5.1] detecting price outliers using iqr method:")
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    price_outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
    print(f"   price outliers detected: {len(price_outliers)} rows")
    print(f"   price range before capping: {df['price'].min():.2f} to {df['price'].max():.2f}")
    
    # cap outliers instead of removing them
    df['price'] = df['price'].clip(lower_bound, upper_bound)
    print(f"   price range after capping: {df['price'].min():.2f} to {df['price'].max():.2f}")
    
    # 5.2 check demand outliers
    print("\n[5.2] checking demand outliers:")
    demand_outliers = df[(df['demand'] < 0) | (df['demand'] > 1)]
    print(f"   demand outliers detected: {len(demand_outliers)} rows")
    df['demand'] = df['demand'].clip(0, 1)
    
    # 5.3 detect competitor price outliers
    print("\n[5.3] detecting competitor price outliers:")
    Q1 = df['competitor_price'].quantile(0.25)
    Q3 = df['competitor_price'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 3 * IQR
    
    competitor_outliers = df[df['competitor_price'] > upper_bound]
    print(f"   competitor price outliers: {len(competitor_outliers)} rows")
    df['competitor_price'] = df['competitor_price'].clip(upper=upper_bound)
    
    print(f"\n   total rows affected by outlier treatment: {original_rows - len(df)}")
    
    return df


# ============================================================================
# STEP 6: DUPLICATES AND GARBAGE VALUE TREATMENTS
# ============================================================================

def step6_remove_duplicates_and_garbage(df):
    """remove duplicate records and garbage values"""
    print("\n" + "=" * 70)
    print("STEP 6: DUPLICATES AND GARBAGE VALUE TREATMENTS")
    print("=" * 70)
    
    original_rows = len(df)
    
    # 6.1 check for exact duplicates
    print("\n[6.1] checking for exact duplicates:")
    exact_duplicates = df.duplicated().sum()
    print(f"   exact duplicate rows: {exact_duplicates}")
    df = df.drop_duplicates()
    
    # 6.2 check for logical duplicates (same trip at same time)
    print("\n[6.2] checking for logical duplicates:")
    logical_cols = ['origin', 'destination', 'timestamp', 'days_to_departure']
    logical_duplicates = df.duplicated(subset=logical_cols).sum()
    print(f"   logical duplicates (same trip): {logical_duplicates}")
    df = df.drop_duplicates(subset=logical_cols, keep='first')
    
    # 6.3 remove garbage values
    print("\n[6.3] removing garbage values:")
    
    garbage_prices = (df['price'] <= 0).sum()
    df = df[df['price'] > 0]
    print(f"   removed {garbage_prices} rows with zero or negative price")
    
    garbage_days = (df['days_to_departure'] < 0).sum()
    df = df[df['days_to_departure'] >= 0]
    print(f"   removed {garbage_days} rows with negative days to departure")
    
    df = df[df['demand'].between(0, 1)]
    
    rows_removed = original_rows - len(df)
    print(f"\n   total rows removed: {rows_removed} ({rows_removed/original_rows*100:.1f}%)")
    print(f"   final dataset size: {len(df)} rows")
    
    return df


# ============================================================================
# STEP 7: NORMALIZATION
# ============================================================================

def step7_normalize_features(df):
    """normalize numerical features for better model performance"""
    print("\n" + "=" * 70)
    print("STEP 7: NORMALIZATION")
    print("=" * 70)
    
    # 7.1 apply standard scaler to numerical features
    print("\n[7.1] applying standard scaler to numerical features:")
    
    features_to_normalize = ['price', 'competitor_price', 'days_to_departure']
    
    print("   before normalization:")
    for col in features_to_normalize:
        print(f"      {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    
    scaler = StandardScaler()
    df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])
    
    print("\n   after normalization:")
    for col in features_to_normalize:
        print(f"      {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}")
    
    # 7.2 save scaler for production use
    print("\n[7.2] saving scaler for production:")
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("   saved scaler to models/scaler.pkl")
    
    return df


# ============================================================================
# STEP 8: ENCODING THE DATA
# ============================================================================

def step8_encode_categorical(df):
    """encode categorical variables for machine learning"""
    print("\n" + "=" * 70)
    print("STEP 8: ENCODING THE DATA")
    print("=" * 70)
    
    # 8.1 one-hot encode origin
    print("\n[8.1] one-hot encoding origin and destination:")
    
    origin_dummies = pd.get_dummies(df['origin'], prefix='origin')
    print(f"   created {origin_dummies.shape[1]} origin dummy columns")
    
    dest_dummies = pd.get_dummies(df['destination'], prefix='dest')
    print(f"   created {dest_dummies.shape[1]} destination dummy columns")
    
    df = pd.concat([df, origin_dummies, dest_dummies], axis=1)
    
    # 8.2 create route combination encoding
    print("\n[8.2] creating route combination encoding:")
    df['route'] = df['origin'] + '_to_' + df['destination']
    route_dummies = pd.get_dummies(df['route'], prefix='route')
    print(f"   created {route_dummies.shape[1]} route dummy columns")
    df = pd.concat([df, route_dummies], axis=1)
    
    # 8.3 drop original text columns
    print("\n[8.3] dropping original text columns:")
    df = df.drop(['origin', 'destination', 'route'], axis=1)
    print("   dropped origin, destination, and route columns")
    
    print(f"\n   final dataframe shape after encoding: {df.shape}")
    
    return df


# ============================================================================
# FEATURE ENGINEERING (DERIVED FEATURES)
# ============================================================================

def engineer_features(df):
    """create additional features that help the model learn patterns"""
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING: CREATING DERIVED FEATURES")
    print("=" * 70)
    
    data = df.copy()
    
    # feature 1: cyclic encoding for hour
    print("\n[feature 1] cyclic encoding for hour:")
    data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
    print("   added hour_sin and hour_cos for cyclic time representation")
    
    # feature 2: seasonal indicators
    print("\n[feature 2] seasonal indicators:")
    data['is_summer'] = data['month'].isin([6, 7, 8]).astype(int)
    data['is_winter_holiday'] = data['month'].isin([12, 1]).astype(int)
    print("   added is_summer and is_winter_holiday flags")
    
    # feature 3: urgency score
    print("\n[feature 3] urgency score:")
    data['urgency'] = 1 / (data['days_to_departure'] + 1)
    print("   added urgency score (inverse of days to departure)")
    
    # feature 4: price competitiveness ratio
    print("\n[feature 4] price competitiveness ratio:")
    data['price_ratio_vs_competitor'] = data['price'] / (data['competitor_price'] + 0.01)
    print("   added price_ratio_vs_competitor")
    
    # feature 5: weekend demand interaction
    print("\n[feature 5] weekend demand interaction:")
    data['weekend_demand_interaction'] = data['is_weekend'] * data['demand']
    print("   added weekend_demand_interaction")
    
    # feature 6: price demand ratio
    print("\n[feature 6] price demand ratio:")
    data['price_demand_ratio'] = data['price'] / (data['demand'] + 0.01)
    print("   added price_demand_ratio")
    
    return data


# ============================================================================
# PREPARE FINAL FEATURES FOR MODEL
# ============================================================================

def prepare_features_for_model(df):
    """prepare final feature matrix X and target vector y"""
    print("\n" + "=" * 70)
    print("PREPARING FEATURES FOR MODEL TRAINING")
    print("=" * 70)
    
    # define the feature columns we want to use
    feature_cols = [
        'price', 'competitor_price', 'days_to_departure', 'is_weekend',
        'hour_sin', 'hour_cos', 'is_summer', 'is_winter_holiday',
        'urgency', 'price_ratio_vs_competitor', 'weekend_demand_interaction',
        'price_demand_ratio'
    ]
    
    # add all one-hot encoded columns
    encoded_cols = [col for col in df.columns if col.startswith(('origin_', 'dest_', 'route_'))]
    feature_cols.extend(encoded_cols)
    
    print(f"total features selected: {len(feature_cols)}")
    
    X = df[feature_cols]
    y = df['demand']
    
    # final check for any missing values
    if X.isnull().sum().sum() > 0:
        print("warning: missing values found, filling with median")
        X = X.fillna(X.median())
    
    print(f"feature matrix shape: {X.shape}")
    print(f"target vector shape: {y.shape}")
    
    return X, y, df


# ============================================================================
# MAIN PIPELINE - RUN ALL STEPS
# ============================================================================

def run_full_pipeline():
    """execute all preprocessing steps in order"""
    print("\n" + "=" * 70)
    print("STARTING COMPLETE DATA PREPROCESSING PIPELINE")
    print("=" * 70)
    
    # step 1: load data
    df = step1_load_data()
    
    # step 2: sanity check
    df = step2_sanity_check(df)
    
    # step 3: exploratory data analysis
    df = step3_exploratory_data_analysis(df)
    
    # step 4: handle missing values
    df = step4_handle_missing_values(df)
    
    # step 5: outliers treatment
    df = step5_handle_outliers(df)
    
    # step 6: remove duplicates and garbage
    df = step6_remove_duplicates_and_garbage(df)
    
    # step 7: normalization
    df = step7_normalize_features(df)
    
    # step 8: encoding
    df = step8_encode_categorical(df)
    
    # feature engineering
    df = engineer_features(df)
    
    # prepare for model
    X, y, df_final = prepare_features_for_model(df)
    
    # save processed data
    os.makedirs('data/processed', exist_ok=True)
    df_final.to_csv('data/processed/clean_data.csv', index=False)
    
    # also save X and y separately for easy loading
    pd.DataFrame(X).to_csv('data/processed/X_features.csv', index=False)
    pd.DataFrame(y, columns=['demand']).to_csv('data/processed/y_target.csv', index=False)
    
    print("\n" + "=" * 70)
    print("PREPROCESSING PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"final dataset saved to: data/processed/clean_data.csv")
    print(f"features saved to: data/processed/X_features.csv")
    print(f"target saved to: data/processed/y_target.csv")
    print(f"eda plots saved to: reports/figures/")
    print(f"scaler saved to: models/scaler.pkl")
    
    return X, y, df_final


if __name__ == "__main__":
    X, y, df = run_full_pipeline()
    
    print("\n" + "=" * 70)
    print("SUMMARY OF FINAL DATASET")
    print("=" * 70)
    print(f"total samples: {len(df)}")
    print(f"total features: {X.shape[1]}")
    print(f"demand range: {y.min():.3f} to {y.max():.3f}")
    print(f"demand mean: {y.mean():.3f}")
