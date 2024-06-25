# Coupon Optimization Strategy

## Overview

The goal of this project is to develop a data-driven coupon strategy to improve the existing approach used by a hypothetical t-shirt selling company. Currently, the company offers a flat 10% discount to all users who abandon their shopping carts. The aim is to create a machine learning model that predicts the optimal coupon variant for each user to maximize conversion rates.

## Project Workflow

This project consists of two main steps:

1. **Data Extraction and Preparation (SQL)**
2. **Model Building and Evaluation (Python)**

### Step 1: Data Extraction and Preparation (SQL)

Using the provided tables (exposures, features, emails, and orders), a training dataset was created. This dataset includes user activity metrics and conversion data.

### Step 2: Model Building and Evaluation (Python)

#### Data Preprocessing

1. **Handling Missing Values**
   - Missing values for features such as `loyalty_points`, `num_past_orders_14d`, and `num_past_clicked_emails_14d` were imputed with zeros to maintain dataset integrity.

2. **Feature Engineering**
   - Additional features were created to better capture user behavior:
     ```python
     df['total_engagement'] = df['num_past_orders_14d'] + df['num_past_clicked_emails_14d']
     df['loyalty_points_per_day'] = df['loyalty_points'] / (df['days_since_signup'] + 1)  # Adding 1 to avoid division by zero
     ```

3. **Log Transformation**
   - Log transformations were applied to highly skewed features to normalize distributions:
     ```python
     skewed_columns = ['loyalty_points']
     for column in skewed_columns:
         df[column] = np.log1p(df[column])
     ```

4. **Handling Outliers**
   - Outliers were capped using the 99th percentile:
     ```python
     def cap_outliers(column):
         upper_cap = df[column].quantile(0.99)
         df[column] = np.where(df[column] > upper_cap, upper_cap, df[column])
     ```

5. **Correlation Analysis**
   - A heatmap was used to analyze correlations between features and identify potential multicollinearity.

#### Model Training

1. **Feature Selection**
   - Selected features based on the correlation analysis and domain knowledge:
     ```python
     X = df.drop(['user_id', 'exposed_at', 'order_conversion', 'revenue'], axis=1)
     y = df['order_conversion']
     ```

2. **Handling Class Imbalance**
   - SMOTE was used to address class imbalance:
     ```python
     smote = SMOTE(random_state=42)
     X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
     ```

3. **Scaling Features**
   - RobustScaler was used to scale the features to handle outliers:
     ```python
     scaler = RobustScaler()
     X_train_scaled = scaler.fit_transform(X_train_resampled)
     X_test_scaled = scaler.transform(X_test)
     ```

4. **Model Training and Evaluation**
   - Various models were trained and evaluated using accuracy, precision, recall, F1 score, and ROC AUC:
     ```python
     models = {
         "Logistic Regression": LogisticRegression(),
         "Random Forest": RandomForestClassifier(random_state=42),
         "Gradient Boosting": GradientBoostingClassifier(random_state=42),
         "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
     }

     for model_name, model in models.items():
         model.fit(X_train_scaled, y_train_resampled)
         y_pred = model.predict(X_test_scaled)
         y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

         accuracy = accuracy_score(y_test, y_pred)
         precision = precision_score(y_test, y_pred)
         recall = recall_score(y_test, y_pred)
         f1 = f1_score(y_test, y_pred)
         roc_auc = roc_auc_score(y_test, y_pred_prob)
     ```
   - The best performing model was chosen for final evaluation and strategy implementation.

#### Strategy Implementation

1. **Predicting Best Coupon**
   - The model was used to predict the best coupon variant for each user:
     ```python
     def predict_best_coupon(df, model, scaler):
         variants = ['variant_zero', 'variant_10_percent', 'variant_20-percent']
         df_features = df.drop(['user_id', 'exposed_at', 'order_conversion', 'revenue'], axis=1)
         df_features_scaled = scaler.transform(df_features)

         variant_probabilities = pd.DataFrame(index=df.index)
         for variant in variants:
             df_variant = df_features.copy()
             for col in variants:
                 df_variant[col] = 0
             df_variant[variant] = 1
             df_variant_scaled = scaler.transform(df_variant)
             variant_probabilities[variant] = model.predict_proba(df_variant_scaled)[:, 1]

         df['best_variant'] = variant_probabilities.idxmax(axis=1)
         return df
     ```

2. **Evaluating Strategy**
   - The effectiveness of the strategy was evaluated by comparing conversion rates:
     ```python
     conversion_with_best_coupon = df[df['best_variant'] != 'variant_zero']['order_conversion'].mean()
     conversion_without_coupon = df[df['best_variant'] == 'variant_zero']['order_conversion'].mean()
     ```

3. **Incorporating Coupon Constraints**
   - Different coupon distribution scenarios were tested to understand their impact:
     ```python
     def apply_coupon_constraints(df, model, scaler, coupon_percentage):
         df_features = df.drop(['user_id', 'exposed_at', 'order_conversion', 'revenue', 'best_variant'], axis=1)
         df_features_scaled = scaler.transform(df_features)

         df['predicted_prob'] = model.predict_proba(df_features_scaled)[:, 1]
         df = df.sort_values('predicted_prob', ascending=False)

         coupon_threshold = int(len(df) * coupon_percentage)
         df['coupon_given'] = 0
         df.iloc[:coupon_threshold, df.columns.get_loc('coupon_given')] = 1

         return df

     coupon_scenarios = [0.1, 0.25, 0.5]
     for scenario in coupon_scenarios:
         df_scenario = apply_coupon_constraints(df.copy(), best_model, scaler, scenario)
         conversion_rate = df_scenario[df_scenario['coupon_given'] == 1]['order_conversion'].mean()
     ```

### Key Results

- **Baseline Conversion Rate:** 4.92%
- **Model Conversion Rate:** 9.97%
- **Conversion Rate with Best Coupon:** 5.33%
- **Conversion Rate without Coupon:** 4.52%

These results highlight the effectiveness of a data-driven coupon strategy in improving conversion rates over the baseline approach.

### Conclusion

This project demonstrates the potential of machine learning in optimizing marketing strategies. By predicting the best coupon variant for each user, the company can significantly increase its conversion rates and make more informed decisions regarding coupon distribution.

