import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from utils import get_project_paths, save_fig

def load_preprocessed_data():
    _, DATA_PATH, _ = get_project_paths()
    df = pd.read_csv(os.path.join(DATA_PATH, "preprocessed.csv"))
    return df

def prepare_features(df):
    numeric_cols = [
        'Application order', 'Admission grade', 'Pre Qual (grade)',
        '1st - enrolled', '1st - evaluations', '1st - approved', 
        '2nd - enrolled', '2nd - evaluations', '2nd - approved',
        'Enroll Age' 
    ]

    categorical_cols = [
        'Marial Status', 'Application mode', 'Course', 'Pre Qual',
        'Daytime/Evening', 'Scholarship', 'Tuition fees up to date',
        'Displaced', 'Gender', 'International',
        "Mom's Qual", "Mom's Occupation",
        "Dad's Qual", "Dad's Occupation"
    ]

    label_map = {"Graduate": 0, "Enrolled": 1, "Dropout": 2}
    df['Target'] = df['Target'].map(label_map)

    X = df[numeric_cols + categorical_cols]
    y = df['Target']
    
    return X, y, numeric_cols, categorical_cols, label_map

def create_preprocessor(numeric_cols, categorical_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ],
        remainder='drop'
    )
    return preprocessor

def train_models(X_train, X_test, y_train, y_test, preprocessor, label_map):
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=400, max_depth=None, random_state=42, 
            class_weight='balanced_subsample', n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}
    prc_scores = {}
    
    for name, model in models.items():
        print(f"   Training {name}...")
        pipe = Pipeline([
            ('prep', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        
        prc_scores[name] = {}
        for i, class_lbl in enumerate(sorted(y_test.unique())):
            if hasattr(pipe.named_steps['model'], "predict_proba"):
                y_scores = pipe.predict_proba(X_test)[:, i]
                y_test_bin = (y_test == class_lbl).astype(int)
                precisions, recalls, thresholds = precision_recall_curve(y_test_bin, y_scores)
                prc_scores[name][class_lbl] = (precisions, recalls, thresholds)

        try:
            if hasattr(pipe.named_steps['model'], "predict_proba"):
                y_proba = pipe.predict_proba(X_test)
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
            else:
                roc_auc = None
        except Exception:
            roc_auc = None
            
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
            'recall_weighted': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc_auc_ovr': roc_auc,
            'pipeline': pipe
        }
    
    return results, prc_scores

def plot_precision_recall_curves(prc_scores, label_map):
    n_models = len(prc_scores)
    if n_models == 0:
        return
    
    n_classes = len(list(prc_scores.values())[0])
    fig, axes = plt.subplots(n_models, n_classes, figsize=(5 * n_classes, 4 * n_models), squeeze=False)
    class_name_map = {v: k for k, v in label_map.items()} if label_map else {}

    for row_i, (model_name, class_dict) in enumerate(prc_scores.items()):
        for col_i, class_lbl in enumerate(sorted(class_dict.keys())):
            precisions, recalls, thresholds = class_dict[class_lbl]
            ax = axes[row_i, col_i]
            ax.plot(thresholds, precisions[:-1], "b--", label="Precision")
            ax.plot(thresholds, recalls[:-1], "g-", label="Recall")
            ax.set_xlabel("Threshold")
            ax.set_title(f"{model_name}: '{class_name_map.get(class_lbl, class_lbl)}'")
            ax.legend(loc="best")
            ax.grid(True)
    
    save_fig('precision_recall_curve')

def plot_confusion_matrices(results, X_test, y_test):
    model_names = list(results.keys())
    y_preds = []
    for model_name in model_names:
        pipe = results[model_name].get('pipeline')
        if pipe is not None:
            y_pred_model = pipe.predict(X_test)
        else:
            y_pred_model = None
        y_preds.append(y_pred_model)

    fig, axes = plt.subplots(1, len(model_names), figsize=(11, 5))
    if len(model_names) == 1:
        axes = [axes]
    
    for idx, (model, y_pred_model) in enumerate(zip(model_names, y_preds)):
        if y_pred_model is not None:
            cm_matrix = confusion_matrix(y_test, y_pred_model)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix)
            disp.plot(ax=axes[idx], cmap='Blues', xticks_rotation=45, colorbar=False)
            axes[idx].set_title(f"{model}")
    
    save_fig('confusion_matrix')

def plot_feature_importances(results, numeric_cols, categorical_cols, preprocessor, X_train):
    top_k = 10
    feature_top_indices = set()
    model_top_data = {}

    feature_names = []
    feature_names.extend(numeric_cols)
    
    for name, trans, cols in preprocessor.transformers_:
        if name == 'cat':
            ohe = trans
            if hasattr(ohe, 'get_feature_names_out'):
                cat_feature_names = list(ohe.get_feature_names_out(cols))
            else:
                cat_feature_names = []
                try:
                    for i, col in enumerate(cols):
                        categories = ohe.categories_[i]
                        cat_feature_names.extend([f"{col}_{cat}" for cat in categories])
                except Exception:
                    pass
            feature_names.extend(cat_feature_names)

    try:
        expected_len = preprocessor.transform(X_train[:1]).shape[1]
        if len(feature_names) != expected_len:
            feature_names = [f"f{i}" for i in range(expected_len)]
    except Exception:
        if len(feature_names) == 0:
            feature_names = [f"f{i}" for i in range(len(numeric_cols))]

    for model_name, info in results.items():
        pipe = info.get('pipeline')
        if pipe is None:
            continue

        model = pipe.named_steps['model']
        if not hasattr(model, 'feature_importances_'):
            continue

        importances = model.feature_importances_
        if importances is None or len(importances) != len(feature_names):
            print(f"Skipping {model_name} due to feature_importances_ shape mismatch.")
            continue

        indices = np.argsort(importances)[-top_k:][::-1]
        feature_top_indices.update(indices)
        model_top_data[model_name] = (indices, importances[indices])

    if not model_top_data:
        print("No models with valid feature_importances_ found.")
        return

    full_indices = sorted(feature_top_indices)
    feature_label_map = {i: feature_names[i] for i in full_indices}
    n_features = len(full_indices)

    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        colors = cm.get_cmap('tab10', len(model_top_data))
    except AttributeError:
        colors = plt.cm.get_cmap('tab10', len(model_top_data))

    bar_width = 0.35
    bar_positions = np.arange(n_features)

    bar_containers = []
    for idx, (model_name, (indices, vals)) in enumerate(model_top_data.items()):
        y = np.zeros(n_features)
        for i, val in zip(indices, vals):
            if i in full_indices:
                y[full_indices.index(i)] = val
        bars = ax.barh(
            bar_positions + idx * bar_width,
            y,
            height=bar_width,
            label=model_name,
            color=colors(idx)
        )
        bar_containers.append((bars, y))

    ax.set_yticks(bar_positions + bar_width * (len(model_top_data) - 1) / 2)
    ax.set_yticklabels([feature_label_map[i] for i in full_indices])
    ax.legend()
    save_fig('top_10_feature_importances_overlap')

    for idx, (bars, y) in enumerate(bar_containers):
        for bar, val in zip(bars, y):
            if val > 0:
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}",
                    va='center',
                    ha='left',
                    fontsize=9
                )

def run_training():
    print("=" * 50)
    print("Model Training")
    print("=" * 50)
    
    print("\n1. Loading preprocessed data...")
    df = load_preprocessed_data()
    print(f"   Data shape: {df.shape}")
    
    print("\n2. Preparing features...")
    X, y, numeric_cols, categorical_cols, label_map = prepare_features(df)
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    print("\n3. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")
    
    print("\n4. Creating preprocessor...")
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    
    print("\n5. Training models...")
    results, prc_scores = train_models(X_train, X_test, y_train, y_test, preprocessor, label_map)
    
    print("\n6. Model Results:")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    print("\n7. Classification Report (RandomForest):")
    rf_pipe = results['RandomForest']['pipeline']
    y_pred_rf = rf_pipe.predict(X_test)
    print(classification_report(y_test, y_pred_rf))
    
    print("\n8. Creating visualizations...")
    
    print("   8.1. Precision-Recall curves...")
    plot_precision_recall_curves(prc_scores, label_map)
    
    print("   8.2. Confusion matrices...")
    plot_confusion_matrices(results, X_test, y_test)
    
    print("   8.3. Feature importances...")
    plot_feature_importances(results, numeric_cols, categorical_cols, preprocessor, X_train)
    
    print("\n" + "=" * 50)
    print("Training completed successfully!")
    print("=" * 50)
    return results

if __name__ == "__main__":
    run_training()