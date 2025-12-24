import flwr as fl
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertModel
from xgboost import Booster, DMatrix
from tqdm import tqdm
import sys

# ======================================
# CONFIG
# ======================================
CLIENT_ID = sys.argv[1]
CSV_PATH = sys.argv[2]

TEXT_COL = "clinical_notes"
TARGET_COL = "Condition"

TABULAR_FEATURES = [
    "Blood_glucose", "HbA1C", "Systolic_BP", "Diastolic_BP",
    "LDL", "HDL", "Triglycerides", "Haemoglobin", "MCV"
]

MODEL_PATH = "models/central_xgboost_model.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[Client {CLIENT_ID}] Loading data from {CSV_PATH}")
print(f"[Client {CLIENT_ID}] Device: {DEVICE}")

# ======================================
# LOAD DATA
# ======================================
df = pd.read_csv(CSV_PATH, engine="python")

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[TARGET_COL])

X_text = df[TEXT_COL]
X_tab = df[TABULAR_FEATURES].astype(float)

# ======================================
# LOAD BERT (FEATURE EXTRACTION)
# ======================================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
bert_model.eval()

def get_bert_embedding(texts):
    embeddings = []
    for text in tqdm(texts, desc=f"[Client {CLIENT_ID}] BERT encoding"):
        tokens = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            outputs = bert_model(**tokens)
        cls_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        embeddings.append(cls_vec)
    return np.array(embeddings)

X_bert = get_bert_embedding(X_text.tolist())
X_final = np.hstack((X_tab.values, X_bert))

# ======================================
# LOAD XGBOOST BOOSTER
# ======================================
booster = Booster()
booster.load_model(MODEL_PATH)

# ======================================
# EVALUATION FUNCTION (FIXED FOR MULTICLASS)
# ======================================
def evaluate_model():
    dmat = DMatrix(X_final)
    probs = booster.predict(dmat)

    # Binary vs Multiclass handling
    if probs.ndim == 1:
        # Binary classification
        preds = (probs > 0.5).astype(int)
    else:
        # Multiclass classification
        preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y, preds)
    return acc

# ======================================
# FLOWER CLIENT
# ======================================
class PretrainedCSVClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return [np.zeros(1, dtype=np.float32)]

    def set_parameters(self, parameters):
        pass

    def fit(self, parameters, config):
        acc = evaluate_model()
        print(f"[Client {CLIENT_ID}] Accuracy: {acc:.4f}")

        return (
            [np.zeros(1, dtype=np.float32)],
            len(X_final),
            {"accuracy": acc}
        )

    def evaluate(self, parameters, config):
        acc = evaluate_model()
        loss = 1.0 - acc
        return loss, len(X_final), {"accuracy": acc}

# ======================================
# START CLIENT
# ======================================
if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=PretrainedCSVClient()
    )
