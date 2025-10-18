import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError("xgboost is required for this script. Please install xgboost.")


def read_csv_fallback(primary_path: str, fallback_path: str) -> pd.DataFrame:
    if os.path.exists(primary_path):
        return pd.read_csv(primary_path)
    if os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)
    raise FileNotFoundError(f"Neither '{primary_path}' nor '{fallback_path}' was found.")


def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    parts = name.split(',')
    if len(parts) < 2:
        return "Unknown"
    right = parts[1].strip()
    title = right.split('.')[0].strip()
    mapping = {
        'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
        'Lady': 'Rare', 'the Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
        'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
        'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
    }
    return mapping.get(title, title)


def extract_surname(name: str) -> str:
    if pd.isna(name):
        return 'UNKNOWN'
    parts = str(name).split(',')
    return parts[0].strip().upper() if parts else 'UNKNOWN'


def cabin_to_deck(cabin: str) -> str:
    if pd.isna(cabin) or not isinstance(cabin, str) or len(cabin) == 0:
        return 'U'  # Unknown
    return cabin[0]


def ticket_prefix(ticket: str) -> str:
    if pd.isna(ticket):
        return 'NONE'
    t = ''.join(ch for ch in str(ticket) if ch.isalnum() or ch in ['/', '.']).upper()
    # Split common separators
    for sep in [' ', '/', '.']:
        if sep in t:
            parts = [p for p in t.split(sep) if p]
            if len(parts) > 1:
                return parts[0]
    # If starts with letters, treat as prefix
    i = 0
    while i < len(t) and t[i].isalpha():
        i += 1
    return t[:i] if i > 0 else 'NONE'


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Basic family features
    out['FamilySize'] = out.get('SibSp', 0) + out.get('Parch', 0) + 1
    out['IsAlone'] = (out['FamilySize'] == 1).astype(int)

    # Family category
    def fam_cat(n):
        if n <= 1:
            return 'Alone'
        if n <= 4:
            return 'Small'
        return 'Large'
    out['FamilyCat'] = out['FamilySize'].apply(fam_cat)

    # Title from name
    out['Title'] = out['Name'].apply(extract_title)
    out['Surname'] = out['Name'].apply(extract_surname)

    # Deck from cabin
    out['Deck'] = out['Cabin'].apply(cabin_to_deck) if 'Cabin' in out.columns else 'U'

    # Ticket frequency and prefix
    if 'Ticket' in out.columns:
        out['TicketFreq'] = out.groupby('Ticket')['Ticket'].transform('count')
        out['TicketPrefix'] = out['Ticket'].apply(ticket_prefix)
    else:
        out['TicketFreq'] = 1
        out['TicketPrefix'] = 'NONE'

    # Fare per person
    if 'Fare' in out.columns:
        denom = out['FamilySize'].replace(0, 1)
        out['FarePerPerson'] = out['Fare'] / denom
    else:
        out['FarePerPerson'] = 0.0

    # Name features
    out['NameLength'] = out['Name'].fillna('').astype(str).str.len()
    out['NameWords'] = out['Name'].fillna('').astype(str).str.split().apply(len)
    # Surname group size
    out['SurnameSize'] = out.groupby('Surname')['Surname'].transform('count')

    # Simple roles
    out['SexNum'] = (out['Sex'] == 'male').astype(int)
    out['IsChild'] = ((out['Age'] < 16).astype(float)).fillna(0).astype(int)
    out['IsMother'] = (
        (out['Sex'] == 'female') &
        (out.get('Parch', 0) > 0) &
        (out['Title'] != 'Miss')
    ).astype(int)

    # Interactions
    if 'Pclass' in out.columns:
        out['AgeTimesClass'] = out['Age'].fillna(out['Age'].median()) * out['Pclass']
        out['FareTimesClass'] = out['Fare'].fillna(out['Fare'].median()) * out['Pclass']
        out['PclassSex'] = out['Pclass'].astype(str) + '_' + out['Sex'].astype(str)

    # Bins (leave raw too; bins help tree models)
    out['AgeBin'] = pd.cut(out['Age'], bins=[-1, 5, 12, 18, 30, 45, 60, 80, 120], labels=False)
    out['FareBin'] = pd.qcut(out['Fare'].fillna(out['Fare'].median()), q=8, duplicates='drop', labels=False)

    return out


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        c for c in [
            'Age', 'Fare', 'Pclass', 'FamilySize', 'IsAlone', 'TicketFreq', 'FarePerPerson',
            'NameLength', 'NameWords', 'SurnameSize', 'SexNum', 'IsChild', 'IsMother', 'AgeTimesClass', 'FareTimesClass'
        ] if c in X.columns
    ]
    categorical_features = [
        c for c in ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix', 'FamilyCat', 'PclassSex', 'AgeBin', 'FareBin'] if c in X.columns
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
            ]), categorical_features)
        ],
        remainder='drop'
    )
    return preprocess


def build_models():
    rf = RandomForestClassifier(
        n_estimators=800,
        max_depth=8,
        min_samples_split=6,
        min_samples_leaf=3,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=900,
        max_depth=4,
        learning_rate=0.035,
        subsample=0.9,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        eval_metric='logloss'
    )
    gbc = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    return rf, xgb_clf, gbc


def generate_oof_predictions(X: pd.DataFrame, y: pd.Series, X_submit: pd.DataFrame, base_pipes: list, cv) -> tuple:
    oof_list = [np.zeros((len(X),), dtype=float) for _ in base_pipes]
    test_pred_list = [np.zeros((len(X_submit),), dtype=float) for _ in base_pipes]

    for fold, (trn_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]

        for i, pipe in enumerate(base_pipes):
            model = clone(pipe)
            model.fit(X_tr, y_tr)
            oof_list[i][val_idx] = model.predict_proba(X_va)[:, 1]
            test_pred_list[i] += model.predict_proba(X_submit)[:, 1] / cv.get_n_splits()

    oof_mat = np.vstack(oof_list).T  # shape: (n_samples, n_models)
    test_mat = np.vstack(test_pred_list).T  # shape: (n_submit, n_models)
    return oof_mat, test_mat


def main():
    parser = argparse.ArgumentParser(description='Titanic submission generator with stacking and tuning (v1)')
    parser.add_argument('--output', type=str, default='submission.csv', help='Tên file CSV output')
    parser.add_argument('--use_single', action='store_true', help='Dùng model đơn tốt nhất thay vì stacking')
    parser.add_argument('--no_tune_threshold', action='store_true', help='Không tune threshold cho stacking (dùng 0.5)')
    parser.add_argument('--folds', type=int, default=7, help='Số folds cho StratifiedKFold')
    args = parser.parse_args()
    # Load data
    train = read_csv_fallback('input/train.csv', 'train.csv')
    test = read_csv_fallback('input/test.csv', 'test.csv')

    # Keep IDs
    test_ids = test['PassengerId'].copy()

    # Feature engineering
    train_fe = engineer_features(train)
    test_fe = engineer_features(test)

    # Define target and features
    y = train_fe['Survived'].astype(int)
    drop_cols = ['Survived']
    if 'PassengerId' in train_fe.columns:
        drop_cols.append('PassengerId')
    if 'PassengerId' in test_fe.columns:
        test_fe = test_fe.drop(columns=['PassengerId'])

    X = train_fe.drop(columns=drop_cols)
    X_submit = test_fe.copy()

    preprocess = build_preprocess(X)
    rf, xgb_clf, gbc = build_models()

    pipe_rf = Pipeline([
        ('preprocess', preprocess),
        ('model', rf)
    ])

    pipe_xgb = Pipeline([
        ('preprocess', preprocess),
        ('model', xgb_clf)
    ])

    pipe_gbc = Pipeline([
        ('preprocess', preprocess),
        ('model', gbc)
    ])

    # Robust CV
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    rf_cv = cross_val_score(pipe_rf, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    xgb_cv = cross_val_score(pipe_xgb, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    gbc_cv = cross_val_score(pipe_gbc, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"RF CV acc: {rf_cv.mean():.4f} (+/- {rf_cv.std()*2:.4f})")
    print(f"XGB CV acc: {xgb_cv.mean():.4f} (+/- {xgb_cv.std()*2:.4f})")
    print(f"GBC CV acc: {gbc_cv.mean():.4f} (+/- {gbc_cv.std()*2:.4f})")

    # Stacking: OOF meta-learning + threshold tuning on OOF
    base_pipes = [pipe_rf, pipe_xgb, pipe_gbc]
    oof_mat, test_mat = generate_oof_predictions(X, y, X_submit, base_pipes, cv)

    meta = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
    meta_cv = cross_val_score(meta, oof_mat, y, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f"Meta (LR) on OOF CV acc: {meta_cv.mean():.4f} (+/- {meta_cv.std()*2:.4f})")

    # Fit meta on full OOF
    meta.fit(oof_mat, y)
    oof_proba = meta.predict_proba(oof_mat)[:, 1]
    test_proba = meta.predict_proba(test_mat)[:, 1]

    # Tune decision threshold using OOF probabilities
    if args.no_tune_threshold:
        best_thr = 0.5
        print("Threshold tuning disabled; using 0.50")
    else:
        best_thr, best_acc = 0.5, 0.0
        for thr in np.linspace(0.35, 0.65, 61):
            pred = (oof_proba >= thr).astype(int)
            acc = (pred == y.values).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
        print(f"Best OOF threshold: {best_thr:.3f} with acc {best_acc:.4f}")

    meta_pred_test = (test_proba >= best_thr).astype(int)

    # Also keep best single model as backup
    best_pipe = pipe_xgb if xgb_cv.mean() >= max(rf_cv.mean(), gbc_cv.mean()) else (pipe_rf if rf_cv.mean() >= gbc_cv.mean() else pipe_gbc)
    best_name = 'XGB' if best_pipe is pipe_xgb else ('RF' if best_pipe is pipe_rf else 'GBC')
    print(f"Best single by CV: {best_name}")
    best_pipe.fit(X, y)
    single_pred_test = best_pipe.predict(X_submit)

    # Save submissions (single file per run)
    if args.use_single:
        out_df = pd.DataFrame({'PassengerId': test_ids, 'Survived': single_pred_test.astype(int)})
        out_df.to_csv(args.output, index=False)
        print(f"Saved best single model submission to {args.output}")
    else:
        out_df = pd.DataFrame({'PassengerId': test_ids, 'Survived': meta_pred_test.astype(int)})
        out_df.to_csv(args.output, index=False)
        print(f"Saved stacked submission to {args.output} (threshold={best_thr:.3f})")


if __name__ == '__main__':
    main()


