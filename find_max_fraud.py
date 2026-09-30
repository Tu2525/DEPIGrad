import joblib
import pandas as pd
import numpy as np
import json

def main():
    with open("feature_config.json", "r") as f:
        config = json.load(f)
    cols = config["feature_names"]
    
    model = joblib.load("fraud_classifier_phase3.pkl")
    transformer = joblib.load("fraud_quantile_transformer.pkl")
    
    print("Searching for the absolute most fraudulent row...")
    chunksize = 20000
    max_prob = 0
    best_raw = None
    
    count = 0
    for chunk in pd.read_csv('augmented_train_qt.csv', chunksize=chunksize):
        if 'isFraud' in chunk.columns:
            f = chunk[chunk['isFraud'] == 1]
            if len(f) > 0:
                features_only = f[cols]
                probs = model.predict_proba(features_only.values)[:, 1]
                
                best_idx = np.argmax(probs)
                if probs[best_idx] > max_prob:
                    max_prob = probs[best_idx]
                    best_scaled_row = features_only.iloc[[best_idx]]
                    best_raw_row = transformer.inverse_transform(best_scaled_row)[0]
                    best_raw = {c: best_raw_row[i] for i, c in enumerate(cols)}
            
            count += 1
            if count > 10 or max_prob > 0.99:
                break
                
    if best_raw:
        print(f"MAX PROB FOUND: {max_prob * 100:.2f}%")
        top_keys = ["TransactionDT", "TransactionAmt", "card1_amt_mean", "card1_amt_std", "V2_te", "V3_te", "C1", "D1", "id_01"]
        print("RAW FRAUD JSON FOR SVELTE:")
        for k in top_keys:
            if k in best_raw:
                print(f"{k}: {best_raw[k]},")
    else:
        print("No frauds found!")

if __name__ == '__main__':
    main()
