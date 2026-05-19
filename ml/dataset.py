from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).parent.parent

DATA_PATH = BASE_DIR /"data" / "raw"/ "SMSSpamCollection"

def load_data():
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["labels","message"])
    return df

def get_features_and_labels():
    df = load_data()
    X= df["message"]
    y= df["labels"]
    return X, y

if __name__ == "__main__":
    
    df=load_data()
    print(df.head())

    print(df["labels"].value_counts())
