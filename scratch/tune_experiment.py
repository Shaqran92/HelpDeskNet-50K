import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

train_df = pd.read_csv("data/processed/train_noisy_symmetric_15pct.csv")
val_df = pd.read_csv("data/processed/val.csv")

y_train = train_df["observed_label"]
y_val = val_df["mapped_label"]

# Test 1: Body only
X_tr_body = train_df["body"].fillna("")
X_val_body = val_df["body"].fillna("")

vec_b = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
X_tr_b = vec_b.fit_transform(X_tr_body)
X_val_b = vec_b.transform(X_val_body)

# Test 2: Subject + Body
X_tr_sb = (train_df["subject"].fillna("") + " " + train_df["body"].fillna("")).str.strip()
X_val_sb = (val_df["subject"].fillna("") + " " + val_df["body"].fillna("")).str.strip()

vec_sb = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, strip_accents="unicode")
X_tr_sb = vec_sb.fit_transform(X_tr_sb)
X_val_sb = vec_sb.transform(X_val_sb)

print("=== BODY ONLY ===")
for C in [0.5, 1.0, 2.0]:
    clf = LinearSVC(C=C, max_iter=3000, random_state=42)
    clf.fit(X_tr_b, y_train)
    preds = clf.predict(X_val_b)
    print(f"LinearSVC C={C:.1f} -> Acc: {accuracy_score(y_val, preds):.4f}, Macro-F1: {f1_score(y_val, preds, average='macro'):.4f}")

for C in [2.0, 5.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    clf.fit(X_tr_b, y_train)
    preds = clf.predict(X_val_b)
    print(f"LogReg    C={C:.1f} -> Acc: {accuracy_score(y_val, preds):.4f}, Macro-F1: {f1_score(y_val, preds, average='macro'):.4f}")

print("\n=== SUBJECT + BODY ===")
for C in [0.5, 1.0, 2.0]:
    clf = LinearSVC(C=C, max_iter=3000, random_state=42)
    clf.fit(X_tr_sb, y_train)
    preds = clf.predict(X_val_sb)
    print(f"LinearSVC C={C:.1f} -> Acc: {accuracy_score(y_val, preds):.4f}, Macro-F1: {f1_score(y_val, preds, average='macro'):.4f}")

for C in [2.0, 5.0, 10.0]:
    clf = LogisticRegression(C=C, max_iter=500, random_state=42)
    clf.fit(X_tr_sb, y_train)
    preds = clf.predict(X_val_sb)
    print(f"LogReg    C={C:.1f} -> Acc: {accuracy_score(y_val, preds):.4f}, Macro-F1: {f1_score(y_val, preds, average='macro'):.4f}")
