import joblib
import pandas as pd
import numpy as np
import json

def main():
    print("Extracting pure RAW fraud data...")
    with open("feature_config.json", "r") as f:
        config = json.load(f)
    cols = config["feature_names"]
    
    transformer = joblib.load("fraud_quantile_transformer.pkl")
    
    frauds_raw = []
    
    for chunk in pd.read_csv('augmented_train_qt.csv', chunksize=20000):
        if 'isFraud' in chunk.columns:
            f = chunk[chunk['isFraud'] == 1]
            if len(f) > 0:
                features_scaled = f[cols]
                raw_values = transformer.inverse_transform(features_scaled)
                
                for r in raw_values:
                    frauds_raw.append(r)
                    if len(frauds_raw) >= 10:
                        break
        if len(frauds_raw) >= 10:
            break
            
    if frauds_raw:
        df_raw = pd.DataFrame(frauds_raw, columns=cols)
        df_raw.to_csv('demo_fraud_data.csv', index=False)
        print("Created demo_fraud_data.csv with 10 exact RAW fraud rows!")
    else:
        print("No frauds extracted.")
        
if __name__ == '__main__':
    main()
