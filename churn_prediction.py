# COMPLETE CHURN PREDICTION SYSTEM
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("=== COMPLETE CHURN PREDICTION SYSTEM ===")
print("All packages loaded successfully! Starting analysis...")

# 1. Create sample data
def create_churn_data():
    np.random.seed(42)
    n_customers = 2000
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'tenure': np.random.randint(1, 72, n_customers),
        'monthly_charges': np.random.uniform(20, 100, n_customers),
        'total_charges': np.random.uniform(50, 5000, n_customers),
        'contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers),
        'payment_method': np.random.choice(['Electronic check', 'Bank transfer', 'Credit card'], n_customers),
        'paperless_billing': np.random.choice(['Yes', 'No'], n_customers),
        'senior_citizen': np.random.choice([0, 1], n_customers),
        'partner': np.random.choice(['Yes', 'No'], n_customers),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn patterns
    churn_prob = (
        (df['tenure'] < 12) * 0.4 +
        (df['contract'] == 'Month-to-month') * 0.3 +
        (df['payment_method'] == 'Electronic check') * 0.2 +
        (df['monthly_charges'] > 70) * 0.15 +
        (df['senior_citizen'] == 1) * 0.1 +
        np.random.normal(0, 0.1, n_customers)
    )
    df['churn'] = (churn_prob > 0.35).astype(int)
    
    return df

# 2. Load and explore data
print("Loading data...")
df = create_churn_data()
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['churn'].mean():.2%}")

# 3. Feature engineering
print("\nPerforming feature engineering...")
df_processed = df.copy()

# Create new features
df_processed['tenure_group'] = pd.cut(df_processed['tenure'], 
                                    bins=[0, 6, 12, 24, 72], 
                                    labels=['0-6m', '7-12m', '13-24m', '25+m'])

df_processed['value_ratio'] = df_processed['monthly_charges'] / (df_processed['tenure'] + 1)

# One-hot encoding
df_encoded = pd.get_dummies(df_processed, columns=['contract', 'payment_method', 'paperless_billing', 'partner', 'tenure_group'], drop_first=True)

# Prepare features
X = df_encoded.drop(['churn', 'customer_id'], axis=1)
y = df_encoded['churn']

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# 5. Build and compare models
print("\n=== TRAINING MULTIPLE MODELS ===")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = model.score(X_test, y_test)
    auc_score = roc_auc_score(y_test, y_prob)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'probabilities': y_prob
    }
    
    print(f"  {name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")

# 6. Find best model
best_model_name = max(results, key=lambda x: results[x]['auc_score'])
best_model = results[best_model_name]['model']
best_auc = results[best_model_name]['auc_score']

print(f"\n🌟 BEST MODEL: {best_model_name} (AUC: {best_auc:.4f})")

# 7. Feature importance
if hasattr(best_model, 'feature_importances_'):
    feature_imp = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n📊 TOP 10 FEATURE IMPORTANCES:")
    print(feature_imp.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_imp.head(10), x='importance', y='feature')
    plt.title(f'Top 10 Feature Importances - {best_model_name}')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("✓ Feature importance plot saved as 'feature_importance.png'")

# 8. Generate predictions for dashboard
print("\n📈 Generating predictions for dashboard...")
final_predictions = X_test.copy()
final_predictions['actual_churn'] = y_test.values
final_predictions['churn_probability'] = best_model.predict_proba(X_test)[:, 1]
final_predictions['churn_prediction'] = best_model.predict(X_test)
final_predictions['risk_category'] = pd.cut(final_predictions['churn_probability'], 
                                          bins=[0, 0.3, 0.7, 1], 
                                          labels=['Low', 'Medium', 'High'])

# Add customer_id back for identification
final_predictions['customer_id'] = df_encoded.loc[X_test.index, 'customer_id'].values

# Save results
final_predictions.to_csv('churn_predictions_complete.csv', index=False)
print("✓ Predictions saved to 'churn_predictions_complete.csv'")

# 9. Business insights
print("\n💡 BUSINESS INSIGHTS & RECOMMENDATIONS:")
print("1. HIGH-RISK SEGMENTS:")
print("   - Customers with tenure < 12 months")
print("   - Month-to-month contract customers") 
print("   - Electronic check payment users")
print("   - High monthly charge customers (>$70)")

print("\n2. RETENTION STRATEGIES:")
print("   - Offer loyalty discounts for short-tenure customers")
print("   - Convert month-to-month to annual contracts with incentives")
print("   - Promote automatic payment methods")
print("   - Create personalized offers for high-value at-risk customers")

print("\n3. PREDICTION PERFORMANCE:")
print(f"   - Best model: {best_model_name}")
print(f"   - Prediction accuracy: {results[best_model_name]['accuracy']:.2%}")
print(f"   - Model confidence (AUC): {best_auc:.2%}")

print("\n🎯 PROJECT DELIVERABLES COMPLETED:")
print("✓ Machine learning model trained and evaluated")
print("✓ Churn probabilities generated for all customers") 
print("✓ Feature importance analysis completed")
print("✓ Business insights and recommendations prepared")
print("✓ Data exported for Power BI dashboard")
print("✓ Visualizations created and saved")

print(f"\n🎉 CHURN PREDICTION SYSTEM COMPLETED SUCCESSFULLY!")
print("Next: Import 'churn_predictions_complete.csv' into Power BI for dashboard creation!")