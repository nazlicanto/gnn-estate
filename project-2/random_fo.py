import os
import joblib
import logging
from datetime import datetime

import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, r2_score


DATA_PATH = './data/alaka.csv'
OUTPUT_DIR = f"rf_shap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# create an output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# set up logging
logging.basicConfig(
    filename=f'{OUTPUT_DIR}/analysis_log.txt', 
    level=logging.INFO, 
    format='%(asctime)s - %(message)s'
)

def engineer_features(df):
    """Engineer features from the raw dataframe."""

    # query related features
    df['query_length'] = df['ilab_query'].str.count('\w+')
    df['query_type'] = df['ilab_query'].apply(lambda x: 'rental' if 'kiralik' in x else 'sale' if 'satilik' in x else 'other')
    df['query_unique_words'] = df['ilab_query'].apply(lambda x: len(set(x.split())))
    df['has_luxury'] = df['ilab_query'].str.contains('luxury|lüks', case=False).astype(int)
    df['has_cheap'] = df['ilab_query'].str.contains('cheap|ucuz', case=False).astype(int)
    df['location'] = df['ilab_query'].str.split('-').str[0]
    df['property_type'] = df['ilab_query'].apply(lambda x: 'apartment' if 'daire' in x else 'house' if 'ev' in x else 'other')

    # time based features
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['hour'] = df['date'].dt.hour

    # interaction-based features
    df['ctr'] = df['sat_click'] / df['attr'].replace(0, np.nan)
    df['purchase_rate'] = df['pur'] / df['sat_click'].replace(0, np.nan)
    df['satisfaction_score'] = (df['sat_click'] + df['sat_pur']) / 2
    df['attr_cont_product'] = df['attr'] * df['cont']
   
    # statistical features
    df['attr_percentile'] = df['attr'].rank(pct=True)
    df['ctr_percentile'] = df['ctr'].rank(pct=True)
    
    # TF-IDF -- SIMULATİON
    tfidf = TfidfVectorizer(max_features=10)
    tfidf_matrix = tfidf.fit_transform(df['ilab_query'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f'tfidf_{i}' for i in range(10)])
    
    # concat features
    df = pd.concat([df, tfidf_df], axis=1)

    # naive approach for filling missing values
    # replace value with mean for numeric & mode for categorical data
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    return df


def get_feature_stats(features):
    """Log feature statistics."""
    # Count features for each original column
    feature_counts = {}
    for feature in features:
        original_feature = feature.split('__')[0] if '__' in feature else feature
        feature_counts[original_feature] = feature_counts.get(original_feature, 0) + 1

    # Log the feature counts
    logging.info("Feature counts after preprocessing:")
    for feature, count in feature_counts.items():
        logging.info(f"{feature}: {count}")

    # Log top 10 feature statistics
    top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logging.info("Top 10 features with most generated features:")
    for feature, count in top_features:
        logging.info(f"{feature}: {count}")


def compute_metrics(estimator, X, y):
    """Compute and log model metrics."""
    # get cross-validation scores
    cv_scores = cross_val_score(estimator, X, y, cv=3, scoring='r2')
    logging.info(f"Cross-validation scores: {cv_scores}")
    logging.info(f"Mean cross-validation score: {cv_scores.mean()}")

    # compute learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=3, scoring='r2', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    # plot results
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, label="Training score")
    plt.plot(train_sizes, test_scores_mean, label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/learning_curve.png')
    plt.close()


def perform_feature_selection(X, y, feature_names, num_est=100, num_feats=10):
    """
    Previously, we were just removing the features with non-zero importance
    Now, a Random Forest model with 100 trees is used as the estimator for RFE.
    RFE selects top 10 features by default, then fit on the pre-processed data.

    Why RFE?: 
        Works well with tree-based models, can analyze the interaction between 
        the features rather then isolated env. features.
    """
    rfe = RFE(
        estimator=RandomForestRegressor(n_estimators=num_est), 
        n_features_to_select=num_feats
    )
    rfe.fit(X, y)

    # get most important features
    selected_feats = [f for f, selected in zip(feature_names, rfe.support_) if selected]
    if not selected_feats:
        logging.warning("RFE selected no features. Check your data or RFE parameters.")
    
    logging.info(f"Selected features after RFE: {selected_feats}")
    return selected_feats


def perform_shap_analysis(estimator, X, feature_names):
    """Perform SHAP analysis and save plots."""
    # SHAP analysis
    explainer = shap.TreeExplainer(estimator.named_steps['regressor'])
    X = estimator.named_steps['preprocessor'].transform(X)
    shap_values = explainer.shap_values(X)

    # SHAP summary plot
    # impact of features on the model output, features ranked by importance
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_summary_plot.png')
    plt.close()

    # SHAP bar plot
    # impact of features on the model output 
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_bar_plot.png')
    plt.close()

    # Calculate and log mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance_df = pd.DataFrame(list(zip(feature_names, mean_abs_shap)), columns=['feature', 'mean_abs_shap_value'])
    feature_importance_df = feature_importance_df.sort_values('mean_abs_shap_value', ascending=False)
    logging.info("\nFeature Importance based on SHAP values:")
    logging.info(feature_importance_df.to_string())

    # Generate SHAP dependence plots for top 5 features
    # how the model output changes as a single feature changes
    top_features = feature_importance_df['feature'].head(5).tolist()
    for feature in top_features:
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(feature, shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(f'{OUTPUT_DIR}/shap_dependence_plot_{feature}.png')
        plt.close()

    # SHAP force plot for a single prediction
    # Visualizes the impact of each feature for a single prediction
    plt.figure(figsize=(20, 3))
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        X[0], 
        feature_names=feature_names, 
        matplotlib=True, 
        show=False
    )
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/shap_force_plot.png')
    plt.close()

    logging.info(f"Analysis complete. Results saved in directory: {OUTPUT_DIR}")


def fit_eval_model(estimator, X_train, y_train, X_test, y_test):
    # Fit the pipeline
    estimator.fit(X_train, y_train)

    # Save the fitted pipeline
    joblib.dump(estimator, f'{OUTPUT_DIR}/rf_pipeline.joblib')
    logging.info(f"Fitted model saved to {OUTPUT_DIR}/rf_pipeline.joblib")

    # Evaluate the model
    y_pred = estimator.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info(f"\nModel Performance on Test Set:")
    logging.info(f"Mean Squared Error: {mse}")
    logging.info(f"R-squared: {r2}")


def main():
    try:
        # Load data
        df = pd.read_csv(DATA_PATH)
        logging.info(f"data loaded from {DATA_PATH}")

        # Feature engineering
        df = engineer_features(df)
        logging.info("feature engineering completed")

        # define initial features
        cols_to_exclude = ['item_id', 'ilab_query']  
        num_feat_selector = make_column_selector(dtype_include=['int64', 'float64'])
        cat_feat_selector = make_column_selector(dtype_include=['object'])

        # Prepare initial features and target
        X = df.drop(columns=cols_to_exclude + ['relevance'])
        y = df['relevance']

        # create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', num_feat_selector),
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feat_selector)
            ],
            remainder='drop'  # drop all unspecified columns
        )

        # preprocess data, split to training and test sets
        X = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Preprocessed X shape: {X_train.shape}")

        # get feature names
        feature_names = (
            num_feat_selector(X) + 
            preprocessor.named_transformers_['cat'].get_feature_names_out(cat_feat_selector(X)).tolist()
        )
        # log feature statistics
        get_feature_stats(feature_names)

        # feature selection with RFE / Recursive Feature Elimination
        selected_feats = perform_feature_selection(X_train, y_train, feature_names)

        # create final pipeline
        # new preprocessor for features selected by RFE
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', [f for f in num_feat_selector(X) if f in selected_feats]),
                ('cat', OneHotEncoder(handle_unknown='ignore'), [f for f in cat_feat_selector(X) if f in selected_feats])
            ]) # one-hot for selected-categorical


        # final model fitting with selected features
        # decided hyperparameters based on experimental results
        rf_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(
                bootstrap=False, max_depth=None, max_features='sqrt',
                min_samples_leaf=1, min_samples_split=2, n_estimators=300, random_state=42
            ))
        ])
        
        # compute and plot metrics
        compute_metrics(rf_pipeline, X_train, y_train)
        logging.info("Learning curve plotted and saved")

        # fit and evaluate the final model
        fit_eval_model(rf_pipeline, X_train, y_train, X_test, y_test)

        # perform SHAP analysis to explain causality
        perform_shap_analysis(rf_pipeline, X_test, selected_feats)

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()