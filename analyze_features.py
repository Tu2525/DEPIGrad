import joblib
import pandas as pd
import json

def main():
    model = joblib.load('fraud_classifier_phase3.pkl')
    df = pd.read_csv('augmented_train_qt.csv', nrows=1)
    
    # Exclude target if it is there
    cols = [c for c in df.columns if c != 'isFraud']
    
    importances = model.feature_importances_
    f_imp = list(zip(cols, importances))
    f_imp.sort(key=lambda x: x[1], reverse=True)
    
    print("Top 15 features:")
    for f, imp in f_imp[:15]:
        print(f"{f}: {imp:.4f}")
        
    # Also let's calculate the medians from a sample of the dataset (e.g. 10000 rows)
    # This will be used for imputing missing values in the app
    print("Calculating medians from sample...")
    sample_df = pd.read_csv('augmented_train_qt.csv', nrows=10000)
    medians = sample_df[cols].median().to_dict()
    with open('feature_medians.json', 'w') as f:
        json.dump(medians, f, indent=2)
    print("Saved feature_medians.json")

if __name__ == '__main__':
    main()
