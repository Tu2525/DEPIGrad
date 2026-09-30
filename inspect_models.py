import joblib
import pandas as pd
import traceback

def main():
    try:
        model = joblib.load("fraud_classifier_phase3.pkl")
        try:
            feats = list(model.feature_names_in_)
        except Exception as e:
            feats = [f"feature_{i}" for i in range(model.n_features_in_)]
            
        with open('model_features.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(feats))
            
        df = pd.read_csv("augmented_train_qt.csv", nrows=0)
        with open('csv_features.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(list(df.columns)))
            
        print("Success! Features exported.")
    except Exception as e:
        print("Error:")
        traceback.print_exc()
        
if __name__ == '__main__':
    main()
