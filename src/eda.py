import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utils import get_project_paths, save_fig

warnings.filterwarnings('ignore')

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

def load_data():
    _, DATA_PATH, _ = get_project_paths()
    df = pd.read_csv(os.path.join(DATA_PATH, "data.csv"), sep=';')
    return df

def rename_columns(df):
    df.columns = [c.replace('\t', ' ').strip() for c in df.columns]
    df = df.rename(columns={
        'Daytime/evening attendance': 'Daytime/Evening',
        'Previous qualification': 'Pre Qual',
        'Previous qualification (grade)': 'Pre Qual (grade)',
        'Mother\'s qualification': 'Mom\'s Qual',
        'Father\'s qualification': 'Dad\'s Qual',
        'Mother\'s occupation': 'Mom\'s Occupation',
        'Father\'s occupation': 'Dad\'s Occupation',
        'Educational special needs': 'Special Needs',
        'Scholarship holder': 'Scholarship',
        'Age at enrollment': 'Enroll Age',
        'Curricular units 1st sem (credited)': '1st - credited',
        'Curricular units 1st sem (enrolled)': '1st - enrolled',
        'Curricular units 1st sem (evaluations)': '1st - evaluations',
        'Curricular units 1st sem (grade)': '1st - grade',
        'Curricular units 1st sem (without evaluations)': '1st - no evaluations',
        'Curricular units 1st sem (approved)': '1st - approved',
        'Curricular units 2nd sem (credited)': '2nd - credited',
        'Curricular units 2nd sem (enrolled)': '2nd - enrolled',
        'Curricular units 2nd sem (evaluations)': '2nd - evaluations',
        'Curricular units 2nd sem (grade)': '2nd - grade',
        'Curricular units 2nd sem (without evaluations)': '2nd - no evaluations',
        'Curricular units 2nd sem (approved)': '2nd - approved',
    })
    return df

def plot_target_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Target'], discrete=True, shrink=0.8)
    plt.title('Target Distribution')
    plt.xlabel('Target')
    save_fig('target_distribution')

def plot_numeric_histograms(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_numeric = [c for c in numeric_cols if c in [
        'Admission grade', 'Pre Qual (grade)', 'Enroll Age',
        '1st - approved', '2nd - grade', '2nd - approved'
    ]]
    if len(selected_numeric) == 0:
        selected_numeric = numeric_cols[:6]

    n = len(selected_numeric)
    cols = 3
    rows = int(np.ceil(n/cols)) if n > 0 else 0
    if n > 0:
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        axes = np.array(axes).reshape(rows, cols)
        for i, col in enumerate(selected_numeric):
            r, c = divmod(i, cols)
            ax = axes[r, c]
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f'{col} distribution')
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        for j in range(i+1, rows*cols):
            r, c = divmod(j, cols)
            axes[r, c].axis('off')
        save_fig('numeric_histograms')

def plot_correlation_heatmap(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr()
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr, cmap='coolwarm', center=0, square=False, linewidths=0.5)
    save_fig('correlation_heatmap')

def plot_target_by_categorical(df):
    cat_candidates = [
        'Marital status', 'Application mode', 'Course', 'Daytime/Evening',
        'Special Needs', 'Debtor', 'Scholarship', 'Displaced', 'Tuition fees up to date'
    ]
    all_cats = [c for c in cat_candidates if c in df.columns]
    n = len(all_cats)
    cols = 3
    rows = int(np.ceil(n / cols)) if n > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()

    legend_handles_labels = None
    for i, col in enumerate(all_cats):
        ct = pd.crosstab(df[col], df['Target'])
        ct = ct.loc[ct.index.sort_values()]
        plot = ct.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[i])
        axes[i].set_title(f'Target count by {col}')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        if legend_handles_labels is None:
            legend_handles_labels = axes[i].get_legend_handles_labels()
        axes[i].get_legend().remove()

    if legend_handles_labels is not None:
        handles, labels = legend_handles_labels
        fig.legend(handles, labels, title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')

    for j in range(len(all_cats), len(axes)):
        axes[j].axis('off')
    save_fig('target_by_categorical_all_count')

def run_eda():
    print("=" * 50)
    print("Exploratory Data Analysis")
    print("=" * 50)
    
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Data shape: {df.shape}")
    print(f"   Data columns: {len(df.columns)}")
    
    print("\n2. Basic statistics...")
    print(df.info())
    print("\n" + str(df.describe()))
    
    print("\n3. Renaming columns...")
    df = rename_columns(df)
    
    print("\n4. Creating visualizations...")
    
    print("   4.1. Target distribution...")
    plot_target_distribution(df)
    
    print("   4.2. Numeric histograms...")
    plot_numeric_histograms(df)
    
    print("   4.3. Correlation heatmap...")
    plot_correlation_heatmap(df)
    
    print("   4.4. Target by categorical features...")
    plot_target_by_categorical(df)
    
    print("\n" + "=" * 50)
    print("EDA completed successfully!")
    print("=" * 50)
    return df

if __name__ == "__main__":
    run_eda()