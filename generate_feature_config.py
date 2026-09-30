import pandas as pd
import json

def main():
    print("Generating feature config from sample...")
    df = pd.read_csv('augmented_train_qt.csv', nrows=10000)
    cols = [c for c in df.columns if c != 'isFraud']
    
    config = {
        'feature_names': cols,
        'medians': df[cols].median().to_dict()
    }

    with open('feature_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Done generating feature config.")

if __name__ == '__main__':
    main()
