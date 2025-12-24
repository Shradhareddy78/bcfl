import numpy as np
import torch
import shap
from transformers import BertTokenizer, BertModel
from xgboost import XGBClassifier

# =========================
# CONFIG
# =========================
MODEL_PATH = "central_xgboost_model.json"
LABEL_CLASSES_PATH = "label_classes.npy"

TABULAR_FEATURES = [
    "Blood_glucose", "HbA1C", "Systolic_BP", "Diastolic_BP",
    "LDL", "HDL", "Triglycerides", "Haemoglobin", "MCV"
]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on:", device)

# =========================
# LOAD LABELS
# =========================
label_classes = np.load(LABEL_CLASSES_PATH, allow_pickle=True)

# =========================
# LOAD MODELS
# =========================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()

model = XGBClassifier()
model.load_model(MODEL_PATH)

# =========================
# BERT EMBEDDING
# =========================
def get_bert_embedding(text):
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

    return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# =========================
# PREDICTION + SHAP (TEXT)
# =========================
def predict_condition_with_explanation(note, tabular_values, top_k=10):

    # ---- Feature construction ----
    bert_vec = get_bert_embedding(note)
    tabular_vec = np.array(tabular_values, dtype=float).reshape(1, -1)
    X = np.hstack((tabular_vec, bert_vec))

    # ---- Prediction ----
    pred_label = model.predict(X)[0]
    pred_class = label_classes[pred_label]
    confidence = np.max(model.predict_proba(X))

    print("\n🎯 PREDICTION RESULT")
    print("Predicted Condition :", pred_class)
    print("Confidence          :", round(confidence, 4))

    # ---- SHAP Explanation ----
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # ✔ Correct multiclass indexing
    # Shape: (1, features, classes)
    class_shap = shap_values[0, :, pred_label]

    # ---- Feature Names ----
    bert_features = [f"BERT_{i}" for i in range(bert_vec.shape[1])]
    feature_names = TABULAR_FEATURES + bert_features

    # ---- Rank features ----
    importance = list(zip(feature_names, class_shap))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n🧠 TOP CONTRIBUTING FEATURES (SHAP - TEXT ONLY)")
    for name, value in importance[:top_k]:
        print(f"{name:25s} → SHAP Impact: {value:+.4f}")

# =========================
# EXAMPLE RUN
# =========================
if __name__ == "__main__":

    sample_note = """
    Patient reports fatigue, frequent urination,
    increased thirst, and elevated blood sugar.
    """

    sample_tabular = [
        180,    # Blood_glucose
        8.2,    # HbA1C
        140,    # Systolic_BP
        90,     # Diastolic_BP
        160,    # LDL
        45,     # HDL
        210,    # Triglycerides
        12.5,   # Haemoglobin
        85      # MCV
    ]

    predict_condition_with_explanation(sample_note, sample_tabular)
