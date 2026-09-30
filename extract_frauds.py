import pandas as pd

def main():
    print("Searching for fraud rows...")
    chunksize = 10000
    frauds = pd.DataFrame()
    for chunk in pd.read_csv('augmented_train_qt.csv', chunksize=chunksize):
        if 'isFraud' in chunk.columns:
            f = chunk[chunk['isFraud'] == 1]
            frauds = pd.concat([frauds, f])
            if len(frauds) >= 5:
                break
                
    if len(frauds) > 0:
        frauds.head(5).to_csv('fraud_samples.csv', index=False)
        print(f"Saved {len(frauds.head(5))} fraud samples to fraud_samples.csv")
        
        # Also print out the top features for the first fraud so we can update the Svelte app
        first_fraud = frauds.iloc[0].to_dict()
        top_keys = ["TransactionDT", "TransactionAmt", "card1_amt_mean", "card1_amt_std", "V2_te", "V3_te", "C1", "D1", "id_01"]
        print("\nJSON for Svelte:")
        for k in top_keys:
            if k in first_fraud:
                print(f"{k}: {first_fraud[k]},")
    else:
        print("No frauds found!")

if __name__ == '__main__':
    main()
