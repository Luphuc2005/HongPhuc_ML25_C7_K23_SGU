import os
import re
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone

try:
    import xgboost as xgb
except Exception as e:
    raise RuntimeError("xgboost is required for this script. Please install xgboost.")

# Optional extras
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except Exception:
    HAS_CAT = False


def read_csv_fallback(primary_path: str, fallback_path: str) -> pd.DataFrame:
    if os.path.exists(primary_path):
        return pd.read_csv(primary_path)
    if os.path.exists(fallback_path):
        return pd.read_csv(fallback_path)
    raise FileNotFoundError(f"Neither '{primary_path}' nor '{fallback_path}' was found.")


def extract_title(name: str) -> str:
    if pd.isna(name):
        return "Unknown"
    title_match = re.search(r' ([A-Za-z]+)\.', name)
    if title_match:
        title = title_match.group(1)
        mapping = {
            'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
            'Lady': 'Rare', 'the Countess': 'Rare', 'Capt': 'Rare', 'Col': 'Rare',
            'Don': 'Rare', 'Dr': 'Rare', 'Major': 'Rare', 'Rev': 'Rare',
            'Sir': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
        }
        return mapping.get(title, title)
    return "Unknown"


def extract_surname(name: str) -> str:
    if pd.isna(name):
        return 'UNKNOWN'
    parts = str(name).split(',')
    return parts[0].strip().upper() if parts else 'UNKNOWN'


def cabin_to_deck(cabin: str) -> str:
    if pd.isna(cabin) or not isinstance(cabin, str) or len(cabin) == 0:
        return 'U'  # Unknown
    return cabin[0]


def num_cabins(cabin: str) -> int:
    if pd.isna(cabin) or not isinstance(cabin, str) or len(cabin) == 0:
        return 0
    return len(cabin.split())


def ticket_prefix(ticket: str) -> str:
    if pd.isna(ticket):
        return 'NONE'
    t = ''.join(ch for ch in str(ticket) if ch.isalnum() or ch in ['/', '.']).upper()
    # Extract first two letters if possible
    match = re.match(r'^([A-Z]{1,2})', t)
    return match.group(1) if match else 'NONE'


def engineer_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    out = df.copy()

    # Handle zero fares as missing
    if 'Fare' in out.columns:
        out.loc[out['Fare'] == 0, 'Fare'] = np.nan

    # Basic family features
    out['FamilySize'] = out.get('SibSp', 0) + out.get('Parch', 0) + 1
    out['IsAlone'] = (out['FamilySize'] == 1).astype(int)

    # Improved family category (4 bins for better granularity)
    def fam_cat(n):
        if n == 1:
            return 'Singleton'
        elif n <= 3:
            return 'Small'
        elif n <= 4:
            return 'Medium'
        else:
            return 'Large'
    out['FamilyCat'] = out['FamilySize'].apply(fam_cat)

    # Title from name (improved with regex)
    out['Title'] = out['Name'].apply(extract_title)
    out['Surname'] = out['Name'].apply(extract_surname)

    # Deck from cabin
    out['Deck'] = out['Cabin'].apply(cabin_to_deck) if 'Cabin' in out.columns else 'U'

    # Number of cabins per passenger
    if 'Cabin' in out.columns:
        out['NumCabins'] = out['Cabin'].apply(num_cabins)
    else:
        out['NumCabins'] = 0

    # Ticket features (prefix + length)
    if 'Ticket' in out.columns:
        out['TicketPrefix'] = out['Ticket'].apply(ticket_prefix)
        out['TicketLen'] = out['Ticket'].astype(str).str.len()
        out['TicketFreq'] = out.groupby('Ticket')['Ticket'].transform('count')
        out['Companions'] = np.maximum(0, out['TicketFreq'] - 1)
    else:
        out['TicketPrefix'] = 'NONE'
        out['TicketLen'] = 0
        out['TicketFreq'] = 1
        out['Companions'] = 0

    # Fare per person and category (10 bins for finer granularity)
    if 'Fare' in out.columns:
        denom = out['FamilySize'].replace(0, 1)
        out['FarePerPerson'] = out['Fare'] / denom
        # Fare category (10 bins)
        out['FareCat'] = pd.qcut(out['Fare'].fillna(out['Fare'].median()), q=10, labels=False, duplicates='drop')
    else:
        out['FarePerPerson'] = 0.0
        out['FareCat'] = 0

    # Name features
    out['NameLength'] = out['Name'].fillna('').astype(str).str.len()
    out['NameWords'] = out['Name'].fillna('').astype(str).str.split().apply(len)
    out['SurnameSize'] = out.groupby('Surname')['Surname'].transform('count')

    # Simple roles
    out['SexNum'] = (out['Sex'] == 'male').astype(int)
    out['IsChild'] = ((out['Age'] < 16).astype(float)).fillna(0).astype(int)
    out['IsMother'] = (
        (out['Sex'] == 'female') &
        (out.get('Parch', 0) > 0) &
        (out['Title'] != 'Miss')
    ).astype(int)

    # Noble flag
    out['Noble'] = (out['Title'] == 'Rare').astype(int)

    # Interactions
    if 'Pclass' in out.columns:
        out['AgeTimesClass'] = out['Age'].fillna(out['Age'].median()) * out['Pclass']
        out['FareTimesClass'] = out['Fare'].fillna(out['Fare'].median()) * out['Pclass']
        out['PclassSex'] = out['Pclass'].astype(str) + '_' + out['Sex'].astype(str)
        out['SexAgeGroup'] = out['Sex'].astype(str) + '_' + pd.cut(out['Age'].fillna(out['Age'].median()), bins=5, labels=False).astype(str)

    # Bins (Age bins + Fare already categorized)
    out['AgeBin'] = pd.cut(out['Age'], bins=[-1, 5, 12, 18, 30, 45, 60, 80, 120], labels=False)

    # Additional features
    out['CabinCount'] = out.groupby('Cabin')['Cabin'].transform('count') if 'Cabin' in out.columns else 1
    out['TicketSurnameMatch'] = (out['TicketFreq'] > 1) & (out['SurnameSize'] > 1)

    # If train, add target-derived features (e.g., family survival rate - but carefully to avoid leak in CV)
    if is_train and 'Survived' in out.columns:
        out['FamilySurvival'] = out.groupby('Surname')['Survived'].transform('mean')
        out['TicketSurvival'] = out.groupby('Ticket')['Survived'].transform('mean')

    return out


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = [
        c for c in [
            'Age', 'Fare', 'Pclass', 'FamilySize', 'IsAlone', 'TicketFreq', 'TicketLen', 'Companions', 'FarePerPerson',
            'NameLength', 'NameWords', 'SurnameSize', 'SexNum', 'IsChild', 'IsMother', 'Noble', 'NumCabins',
            'AgeTimesClass', 'FareTimesClass', 'CabinCount', 'TicketSurnameMatch', 'FamilySurvival', 'TicketSurvival'
        ] if c in X.columns
    ]
    categorical_features = [
        c for c in ['Sex', 'Embarked', 'Title', 'Deck', 'TicketPrefix', 'FamilyCat', 'PclassSex', 
                    'AgeBin', 'FareCat', 'SexAgeGroup'] if c in X.columns
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numeric_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ]), categorical_features)
        ],
        remainder='drop'
    )
    return preprocess


def build_models():
    rf = RandomForestClassifier(
        n_estimators=1000,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        n_jobs=-1,
        random_state=42
    )

    xgb_clf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        tree_method='hist',
        eval_metric='logloss'
    )
    gbc = GradientBoostingClassifier(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=4,
        random_state=42
    )
    models = {'rf': rf, 'xgb': xgb_clf, 'gbc': gbc}
    if HAS_LGB:
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.02,
            max_depth=-1,
            num_leaves=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        models['lgb'] = lgb_clf
    if HAS_CAT:
        cat_clf = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.02,
            depth=7,
            l2_leaf_reg=3.0,
            loss_function='Logloss',
            eval_metric='AUC',
            random_state=42,
            verbose=False,
            thread_count=-1
        )
        models['cat'] = cat_clf
    return models


def postprocess_predictions(df_submit: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    pred = (proba >= 0.5).astype(int)
    if 'Sex' in df_submit.columns:
        female_idx = (df_submit['Sex'] == 'female').values
        close_neg = (proba < 0.55) & (proba >= 0.35)
        pred[female_idx & close_neg] = 1
    if 'IsChild' in df_submit.columns:
        child_idx = (df_submit['IsChild'] == 1).values
        close_neg = (proba < 0.60) & (proba >= 0.35)
        pred[child_idx & close_neg] = 1
    if 'Pclass' in df_submit.columns and 'Sex' in df_submit.columns:
        male_3rd = (df_submit['Sex'] == 'male') & (df_submit['Pclass'] == 3)
        close_pos = (proba < 0.65) & (proba >= 0.40)
        pred[male_3rd & close_pos] = 0
        female_1st = (df_submit['Sex'] == 'female') & (df_submit['Pclass'] == 1)
        close_neg = (proba < 0.60) & (proba >= 0.45)
        pred[female_1st & close_neg] = 1
    return pred


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

    oof_mat = np.vstack(oof_list).T
    test_mat = np.vstack(test_pred_list).T
    return oof_mat, test_mat


def main():
    parser = argparse.ArgumentParser(description='Titanic submission generator with stacking and tuning')
    parser.add_argument('--output', type=str, default='submission.csv', help='Tên file CSV output')
    parser.add_argument('--use_single', action='store_true', help='Dùng model đơn tốt nhất thay vì stacking')
    parser.add_argument('--no_tune_threshold', action='store_true', help='Không tune threshold cho stacking (dùng 0.5)')
    parser.add_argument('--folds', type=int, default=5, help='Số folds cho StratifiedKFold')
    args = parser.parse_args()

    # Load data
    train = read_csv_fallback('input/train.csv', 'train.csv')
    test = read_csv_fallback('input/test.csv', 'test.csv')

    # Keep IDs
    test_ids = test['PassengerId'].copy()

    # Feature engineering (improved, with train flag for target-derived features)
    train_fe = engineer_features(train, is_train=True)
    test_fe = engineer_features(test, is_train=False)

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
    models = build_models()

    pipes = {}
    for key, mdl in models.items():
        pipes[key] = Pipeline([
            ('preprocess', preprocess),
            ('model', mdl)
        ])

    # CV and tuning (increased n_iter for better tuning)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    cv_scores = {}
    for key, pipe in pipes.items():
        if key in ['xgb', 'lgb', 'cat', 'rf', 'gbc']:  # Tune all models
            if key == 'rf':
                param_dist = {
                    'model__n_estimators': [800, 1000, 1200],
                    'model__max_depth': [8, 10, 12]
                }
            elif key == 'gbc':
                param_dist = {
                    'model__n_estimators': [400, 500, 600],
                    'model__learning_rate': [0.03, 0.04, 0.05]
                }
            else:  # Boosting
                param_dist = {
                    'model__n_estimators': [800, 1000, 1200],
                    'model__learning_rate': [0.02, 0.03, 0.04]
                }
            tuner = RandomizedSearchCV(pipe, param_dist, n_iter=10, cv=3, scoring='accuracy', random_state=42, n_jobs=-1)  # Increased n_iter
            tuner.fit(X, y)
            scores = cross_val_score(tuner.best_estimator_, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
            pipes[key] = tuner.best_estimator_
        else:
            scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        cv_scores[key] = scores
        print(f"{key.upper()} CV acc: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")

    # Stacking with top models
    ranked = sorted(cv_scores.items(), key=lambda kv: kv[1].mean(), reverse=True)
    base_keys = [k for k, _ in ranked[:5]]
    base_pipes = [pipes[k] for k in base_keys]
    oof_mat, test_mat = generate_oof_predictions(X, y, X_submit, base_pipes, cv)

    # Meta-learner with early stopping (split for validation)
    X_temp, X_val, y_temp, y_val = train_test_split(oof_mat, y, test_size=0.2, stratify=y, random_state=42)
    meta = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, random_state=42, n_jobs=-1)
    meta.fit(X_temp, y_temp, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=False)
    oof_proba = meta.predict_proba(oof_mat)[:, 1]
    test_proba = meta.predict_proba(test_mat)[:, 1]

    # Threshold tuning (finer)
    if args.no_tune_threshold:
        best_thr = 0.5
        print("Threshold tuning disabled; using 0.50")
    else:
        best_thr, best_acc = 0.5, 0.0
        for thr in np.linspace(0.30, 0.70, 101):
            pred = (oof_proba >= thr).astype(int)
            acc = (pred == y.values).mean()
            if acc > best_acc:
                best_acc = acc
                best_thr = thr
        print(f"Best OOF threshold: {best_thr:.3f} with acc {best_acc:.4f}")

    meta_pred_test = (test_proba >= best_thr).astype(int)

    # Best single as backup
    best_key = max(cv_scores.keys(), key=lambda k: cv_scores[k].mean())
    best_pipe = pipes[best_key]
    print(f"Best single by CV: {best_key.upper()}")
    best_pipe.fit(X, y)
    single_pred_test = best_pipe.predict(X_submit)

    # Save
    if args.use_single:
        out_df = pd.DataFrame({'PassengerId': test_ids, 'Survived': single_pred_test.astype(int)})
        out_df.to_csv(args.output, index=False)
        print(f"Saved best single model submission to {args.output}")
    else:
        submit_features = engineer_features(read_csv_fallback('input/test.csv', 'test.csv'), is_train=False)
        if 'PassengerId' in submit_features.columns:
            submit_features = submit_features.drop(columns=['PassengerId'])
        tuned_pred = postprocess_predictions(submit_features, test_proba)
        out_df = pd.DataFrame({'PassengerId': test_ids, 'Survived': tuned_pred.astype(int)})
        out_df.to_csv(args.output, index=False)
        print(f"Saved stacked submission to {args.output} (threshold={best_thr:.3f} + postprocess)")


if __name__ == '__main__':
    main()