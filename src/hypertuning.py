import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    make_scorer, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_sample_weight
from utils import get_project_paths, save_fig

def load_preprocessed_data():
    _, DATA_PATH, _ = get_project_paths()
    df = pd.read_csv(os.path.join(DATA_PATH, "preprocessed.csv"))
    return df

def create_engineered_features(df):
    df = df.copy()
    
    for sem in ['1st', '2nd']:
        en_col = f'{sem} - enrolled'
        ap_col = f'{sem} - approved'
        ev_col = f'{sem} - evaluations'
        gr_col = f'{sem} - grade'

        enrolled = df[en_col] if en_col in df.columns else pd.Series(0, index=df.index)
        approved = df[ap_col] if ap_col in df.columns else pd.Series(0, index=df.index)
        evaluations = df[ev_col] if ev_col in df.columns else pd.Series(0, index=df.index)
        grade = df[gr_col] if gr_col in df.columns else pd.Series(0, index=df.index)

        df[f'{sem}_approval_rate'] = np.where(enrolled > 0, approved / enrolled, 0.0)
        df[f'{sem}_evaluation_rate'] = np.where(enrolled > 0, evaluations / enrolled, 0.0)
        df[f'{sem}_avg_grade'] = np.where(evaluations > 0, grade / evaluations, 0.0)

    df['delta_approval_rate'] = df.get('2nd_approval_rate', 0.0) - df.get('1st_approval_rate', 0.0)
    df['delta_avg_grade'] = df.get('2nd_avg_grade', 0.0) - df.get('1st_avg_grade', 0.0)

    if 'Enroll Age' in df.columns:
        df['Enroll Age'] = pd.to_numeric(df['Enroll Age'], errors='coerce')
        df['age_group'] = pd.cut(df['Enroll Age'],
                                     bins=[-1, 20, 24, 30, 200],
                                     labels=[0, 1, 2, 3]).astype(float).fillna(3).astype(int)

    return df

def visualize_engineered_features(df):
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    new_feature_cols = [
        '1st_approval_rate', '2nd_approval_rate',
        '1st_evaluation_rate', '2nd_evaluation_rate',
        '1st_avg_grade', '2nd_avg_grade',
        'delta_approval_rate', 'delta_avg_grade',
        'age_group'
    ]

    fig, axs = plt.subplots(3, 3, figsize=(16, 12))
    axs = axs.flatten()

    for i, col in enumerate(new_feature_cols):
        if col not in df.columns:
            axs[i].axis('off')
            continue
            
        ax = axs[i]
        unique_targets = df['Target'].unique()
        if df[col].nunique() < 20:
            value_counts = df.groupby('Target')[col].value_counts().unstack(fill_value=0).sort_index(axis=1)
            value_counts.T.plot(kind='bar', ax=ax)
            ax.legend(title='Target')
        else:
            for target in unique_targets:
                df[df['Target'] == target][col].hist(
                    bins=30, alpha=0.6, ax=ax, label=str(target)
                )
            ax.legend(title='Target')
        ax.set_title(col)
    
    save_fig('eng_feats')

def prepare_features(df):
    numeric_cols = [
        'Application order', 'Admission grade', 'Pre Qual (grade)',
        '1st - enrolled', '1st - evaluations', '1st - approved', 
        '2nd - enrolled', '2nd - evaluations', '2nd - approved',
        '1st_avg_grade', '1st_approval_rate',
        '2nd_avg_grade', '2nd_approval_rate',
        'delta_approval_rate', 'delta_avg_grade'    
    ]

    categorical_cols = [
        'Marial Status', 'Application mode', 'Course', 'Pre Qual',
        'Nationality', 'Daytime/Evening', 'Scholarship', 'Tuition fees up to date',
        'Displaced', 'Gender', 'International', 'age_group',
        "Mom's Qual", "Mom's Occupation",
        "Dad's Qual", "Dad's Occupation",
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

def create_rf_pipeline(preprocessor):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])

def create_gb_pipeline(preprocessor):
    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=42))
    ])

def hyperparameter_tuning_rf(X_train, y_train, preprocessor, n_iter=50):
    print("   Tuning RandomForest...")
    
    rf_param_grid = {
        'n_estimators': [300, 400, 500],
        'max_depth': [20, 25, 30],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [2, 4, 6],
        'max_features': ['sqrt', 'log2'],
        'class_weight': ['balanced_subsample'],
        'criterion': ['gini', 'entropy']
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='macro')
    
    rf_search = RandomizedSearchCV(
        create_rf_pipeline(preprocessor),
        param_distributions={'classifier__' + k: v for k, v in rf_param_grid.items()},
        n_iter=n_iter,
        scoring={'f1_macro': f1_scorer, 'accuracy': 'accuracy'}, 
        refit='accuracy',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    rf_search.fit(X_train, y_train)
    print(f"   Best RF CV Score: {rf_search.best_score_:.4f}")
    print(f"   Best RF Params: {rf_search.best_params_}")
    
    return rf_search

def hyperparameter_tuning_gb(X_train, y_train, preprocessor, n_iter=50):
    print("   Tuning GradientBoosting...")
    
    gb_param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.15],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [10, 20, 30],
        'min_samples_leaf': [5, 10, 15],
        'subsample': [0.8, 0.9, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_scorer = make_scorer(f1_score, average='macro')
    sample_weights = compute_sample_weight('balanced', y_train)
    
    gb_search = RandomizedSearchCV(
        create_gb_pipeline(preprocessor),
        param_distributions={'classifier__' + k: v for k, v in gb_param_grid.items()},
        n_iter=n_iter,
        scoring={'f1_macro': f1_scorer, 'accuracy': 'accuracy'}, 
        refit='accuracy',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=True
    )

    gb_search.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    print(f"   Best GB CV Score: {gb_search.best_score_:.4f}")
    print(f"   Best GB Params: {gb_search.best_params_}")
    
    return gb_search

def plot_cv_results(rf_search, gb_search):
    rf_cv_results = rf_search.cv_results_
    mean_train_acc = rf_cv_results['mean_train_accuracy']
    mean_val_acc = rf_cv_results['mean_test_accuracy']
    mean_train_loss = 1 - mean_train_acc

    plt.figure(figsize=(9, 5))
    plt.plot(mean_train_acc, label='Train Accuracy', marker='o')
    plt.plot(mean_val_acc, label='Validation Accuracy', marker='s')
    plt.plot(mean_train_loss, label='Train Loss', marker='x')
    plt.xlabel('Hyperparameter Search Iteration')
    plt.title('Random Forest')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    gb_cv_results = gb_search.cv_results_
    gb_mean_train_acc = gb_cv_results['mean_train_accuracy']
    gb_mean_val_acc = gb_cv_results['mean_test_accuracy']
    gb_mean_train_loss = 1 - gb_mean_train_acc

    plt.figure(figsize=(8, 5))
    plt.plot(gb_mean_train_acc, label='Train Accuracy', marker='o')
    plt.plot(gb_mean_val_acc, label='Validation Accuracy', marker='s')
    plt.plot(gb_mean_train_loss, label='Train Loss', marker='x')
    plt.xlabel('Hyperparameter Search Iteration')
    plt.title('Gradient Boosting')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1_Macro': f1_score(y_test, y_pred, average='macro'),
        'F1_Weighted': f1_score(y_test, y_pred, average='weighted'),
        'Precision_Macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Recall_Macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'ROC_AUC': roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    }
    return results

def plot_confusion_matrices_hypetuned(rf_model, gb_model, X_test, y_test):
    model_names = ['RandomForest', 'GradientBoosting']
    models = [rf_model, gb_model]
    y_preds = [model.predict(X_test) for model in models]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for idx, (name, y_pred) in enumerate(zip(model_names, y_preds)):
        cm_matrix = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm_matrix, display_labels=np.unique(y_test))
        disp.plot(ax=axes[idx], cmap='Blues', xticks_rotation=45, colorbar=False)
        axes[idx].set_title(f"{name}")
    save_fig("confusion_matrix_hypetuned")

def plot_feature_importances_hypetuned(rf_model, gb_model, numeric_cols, categorical_cols, preprocessor):
    cat_names = rf_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numeric_cols + list(cat_names)

    rf_importances = rf_model.named_steps['classifier'].feature_importances_
    gb_importances = gb_model.named_steps['classifier'].feature_importances_

    n_features = min(len(feature_names), len(rf_importances), len(gb_importances))
    feature_names = feature_names[:n_features]
    rf_importances = rf_importances[:n_features]
    gb_importances = gb_importances[:n_features]

    rf_top_idx = np.argsort(rf_importances)[-10:][::-1]
    gb_top_idx = np.argsort(gb_importances)[-10:][::-1]
    all_top_idx = sorted(set(rf_top_idx) | set(gb_top_idx), key=lambda i: -(rf_importances[i] + gb_importances[i]))
    all_top_idx = all_top_idx[:10]

    top_features = [feature_names[i] for i in all_top_idx][::-1]
    rf_top_vals = [rf_importances[i] for i in all_top_idx][::-1]
    gb_top_vals = [gb_importances[i] for i in all_top_idx][::-1]
    y = np.arange(len(top_features))

    width = 0.35
    fig, ax = plt.subplots(figsize=(13, 8))
    bars1 = ax.barh(y - width/2, rf_top_vals, height=width, label='RandomForest')
    bars2 = ax.barh(y + width/2, gb_top_vals, height=width, label='GradientBoosting')

    for bars, vals in zip([bars1, bars2], [rf_top_vals, gb_top_vals]):
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}",
                    va='center', ha='left',
                    fontsize=9, color='black', clip_on=False
                )

    ax.set_ylabel(None)
    ax.set_xlabel(None)
    ax.set_yticks(y)
    ax.set_yticklabels(top_features)
    ax.legend()
    save_fig("top10_feature_importances_enhanced")

def run_hypertuning(n_iter=50):
    print("=" * 50)
    print("Feature Engineering & Hyperparameter Tuning")
    print("=" * 50)
    
    print("\n1. Loading preprocessed data...")
    df = load_preprocessed_data()
    print(f"   Data shape: {df.shape}")
    
    print("\n2. Creating engineered features...")
    df = create_engineered_features(df)
    print(f"   Data shape after feature engineering: {df.shape}")
    
    print("\n3. Saving engineered features...")
    _, DATA_PATH, _ = get_project_paths()
    output_path = os.path.join(DATA_PATH, "engineered_features.csv")
    df.to_csv(output_path, index=False)
    print(f"   Engineered features saved to {output_path}")
    
    print("\n4. Visualizing engineered features...")
    visualize_engineered_features(df)
    
    print("\n5. Preparing features...")
    X, y, numeric_cols, categorical_cols, label_map = prepare_features(df)
    print(f"   Numeric features: {len(numeric_cols)}")
    print(f"   Categorical features: {len(categorical_cols)}")
    
    print("\n6. Creating train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"   Train size: {len(X_train)}")
    print(f"   Test size: {len(X_test)}")
    
    print("\n7. Creating preprocessor...")
    preprocessor = create_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(X_train)
    
    print("\n8. Performing hyperparameter tuning...")
    print(f"   Number of iterations: {n_iter}")
    rf_search = hyperparameter_tuning_rf(X_train, y_train, preprocessor, n_iter=n_iter)
    gb_search = hyperparameter_tuning_gb(X_train, y_train, preprocessor, n_iter=n_iter)
    
    print("\n9. Plotting CV results...")
    plot_cv_results(rf_search, gb_search)
    
    print("\n10. Evaluating models on test set...")
    rf_results = evaluate_model(rf_search.best_estimator_, X_test, y_test, 'RandomForest')
    gb_results = evaluate_model(gb_search.best_estimator_, X_test, y_test, 'GradientBoosting')
    
    print("\n11. Creating ensemble model...")
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf_search.best_estimator_),
            ('gb', gb_search.best_estimator_)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_results = evaluate_model(ensemble, X_test, y_test, 'Ensemble')
    
    print("\n12. Model Comparison:")
    comparison_df = pd.DataFrame([rf_results, gb_results, ensemble_results]).set_index('Model')
    print(comparison_df)
    
    print("\n13. Creating visualizations...")
    
    print("   13.1. Confusion matrices...")
    plot_confusion_matrices_hypetuned(rf_search.best_estimator_, gb_search.best_estimator_, X_test, y_test)
    
    print("   13.2. Feature importances...")
    plot_feature_importances_hypetuned(rf_search.best_estimator_, gb_search.best_estimator_, 
                                      numeric_cols, categorical_cols, preprocessor)
    
    print("\n14. Overfitting Analysis:")
    for name, search in [('RandomForest', rf_search), ('GradientBoosting', gb_search)]:
        cv_score = search.best_score_
        test_score = comparison_df.loc[name, 'Accuracy']
        diff = cv_score - test_score
        print(f"{name}:")
        print(f"  CV Score: {cv_score:.4f}")
        print(f"  Test Score: {test_score:.4f}")
        print(f"  Difference: {diff:.4f} {'(HIGH OVERFITTING!)' if diff > 0.05 else '(acceptable)'}")
    
    print("\n" + "=" * 50)
    print("Hyperparameter tuning completed successfully!")
    print("=" * 50)
    return rf_search, gb_search, ensemble

if __name__ == "__main__":
    run_hypertuning()