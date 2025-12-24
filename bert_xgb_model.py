import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel
from xgboost import XGBClassifier

DATA_PATH = "healthmarkers.csv"

TEXT_COL = "clinical_notes"
TARGET_COL = "Condition"
TABULAR_FEATURES = [
    "Blood_glucose", "HbA1C", "Systolic_BP", "Diastolic_BP",
    "LDL", "HDL", "Triglycerides", "Haemoglobin", "MCV"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

df = pd.read_csv(DATA_PATH, engine="python")

label_encoder = LabelEncoder()
df[TARGET_COL] = label_encoder.fit_transform(df[TARGET_COL])
np.save("label_classes.npy", label_encoder.classes_)

X_text = df[TEXT_COL]
X_tab = df[TABULAR_FEATURES].astype(float)
y = df[TARGET_COL]

X_train_text, X_test_text, X_train_tab, X_test_tab, y_train, y_test = train_test_split(
    X_text, X_tab, y, test_size=0.2, random_state=42
)

print("Total Training Samples:", len(X_train_text))
print("Total Testing Samples:", len(X_test_text))

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

def get_bert_embedding(texts):
    embeddings = []
    for text in tqdm(texts, desc="Encoding clinical notes using BERT"):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = bert_model(**tokens)
        cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        embeddings.append(cls_vec)
    return np.array(embeddings)

X_train_bert = get_bert_embedding(X_train_text.tolist())
X_test_bert = get_bert_embedding(X_test_text.tolist())

print("\nBERT Encoding Complete ✔")

X_train_final = np.hstack((X_train_tab.values, X_train_bert))
X_test_final = np.hstack((X_test_tab.values, X_test_bert))

print("Final Feature Shape:", X_train_final.shape)


model = XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss"
)

model.fit(X_train_final, y_train)

# Save model + metadata
model.save_model("central_xgboost_model.json")
np.save("bert_feature_indices.npy", TABULAR_FEATURES)

print("\nModel Saved Successfully ✔")


preds = model.predict(X_test_final)

print("\n🎯 MODEL RESULTS 🎯")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
