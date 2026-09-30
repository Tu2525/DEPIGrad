import joblib
import pandas as pd
import json

def main():
    transformer = joblib.load("fraud_quantile_transformer.pkl")
    with open("feature_config.json", "r") as f:
        config = json.load(f)
        
    cols = config["feature_names"]
    
    # 1. Compute raw fraud sample
    fraud_df = pd.read_csv("fraud_samples.csv")
    raw_frau = transformer.inverse_transform(fraud_df[cols])
    raw_frau_df = pd.DataFrame(raw_frau, columns=cols)
    raw_first = raw_frau_df.iloc[0].to_dict()
    
    top_keys = ["TransactionDT", "TransactionAmt", "card1_amt_mean", "card1_amt_std", "V2_te", "V3_te", "C1", "D1", "id_01"]
    print("RAW FRAUD JSON FOR SVELTE:")
    for k in top_keys:
        if k in raw_first:
            print(f"{k}: {raw_first[k]},")
            
if __name__ == "__main__":
    main()
