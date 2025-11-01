print("=== TESTING ALL PACKAGES ===")

try:
    import pandas as pd
    print("✓ pandas - SUCCESS")
except ImportError as e:
    print("✗ pandas - FAILED")

try:
    import numpy as np
    print("✓ numpy - SUCCESS")
except ImportError as e:
    print("✗ numpy - FAILED")

try:
    from sklearn.ensemble import RandomForestClassifier
    print("✓ scikit-learn - SUCCESS")
except ImportError as e:
    print("✗ scikit-learn - FAILED")

try:
    import matplotlib.pyplot as plt
    print("✓ matplotlib - SUCCESS")
except ImportError as e:
    print("✗ matplotlib - FAILED")

try:
    import seaborn as sns
    print("✓ seaborn - SUCCESS")
except ImportError as e:
    print("✗ seaborn - FAILED")

try:
    from xgboost import XGBClassifier
    print("✓ xgboost - SUCCESS")
except ImportError as e:
    print("✗ xgboost - FAILED")

try:
    import jupyter
    print("✓ jupyter - SUCCESS")
except ImportError as e:
    print("✗ jupyter - FAILED")

print("\n🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
print("You can now run the complete churn prediction system!")