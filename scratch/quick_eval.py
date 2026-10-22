import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

train = pd.read_csv("data/processed/train_noisy_symmetric_15pct.csv")
val = pd.read_csv("data/processed/val.csv")

y_train = train["observed_label"]
y_val = val["mapped_label"]

for name, tr_text, val_text in [
    ("Body only", train["body"].fillna(""), val["body"].fillna("")),
    ("Subject + Body", (train["subject"].fillna("") + " " + train["body"].fillna("")).str.strip(), (val["subject"].fillna("") + " " + val["body"].fillna("")).str.strip()),
]:
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
    X_tr = vec.fit_transform(tr_text)
    X_v = vec.transform(val_text)
    
    for C in [0.2, 0.5, 1.0, 2.0]:
        svc = LinearSVC(C=C, random_state=42, dual="auto")
        svc.fit(X_tr, y_train)
        pred = svc.predict(X_v)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro")
        print(f"{name:15s} | SVC C={C:3.1f} -> Acc: {acc:.4f}, Macro-F1: {f1:.4f}")
    
    for C in [1.0, 3.0, 5.0, 10.0]:
        lr = LogisticRegression(C=C, max_iter=400, tol=1e-3, random_state=42)
        lr.fit(X_tr, y_train)
        pred = lr.predict(X_v)
        acc = accuracy_score(y_val, pred)
        f1 = f1_score(y_val, pred, average="macro")
        print(f"{name:15s} | LR  C={C:4.1f} -> Acc: {acc:.4f}, Macro-F1: {f1:.4f}")
