import pandas as pd
from bert_xgb_model import BERTXGBModel

df = pd.read_csv("data/hospitalA.csv")
model = BERTXGBModel()
model.train(df)

preds = model.predict(df)
print("Predictions:", preds)
