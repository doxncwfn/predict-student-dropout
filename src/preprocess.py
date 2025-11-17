import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from utils import get_project_paths, save_fig

warnings.filterwarnings('ignore')

def marial(x):
    if x == 1:
        return 1
    else:
        return 0

def app_mode(x):
    standard = {1, 5, 16, 17, 18}
    ordinance = {2, 10, 26, 27}
    mature = {7, 39, 44, 53}
    transfer = {42, 43, 51, 57}
    international = {15}

    if x in standard:
        return 0
    elif x in ordinance:
        return 1
    elif x in mature:
        return 2
    elif x in transfer:
        return 3
    elif x in international:
        return 4
    else:
        return 5
    
def course(x):
    eng_tech = {33, 9119}
    design_media = {171, 9070, 9670, 9773}
    agri_animal = {9003, 9130, 9085}
    business_mgmt = {9147, 9991, 9254}
    health = {9500, 9556}
    social_edu = {9238, 8014, 9853}

    if x in eng_tech:
        return 0
    elif x in design_media:
        return 1
    elif x in agri_animal:
        return 2
    elif x in business_mgmt:
        return 3
    elif x in health:
        return 4
    elif x in social_edu:
        return 5
    else:
        return 6

def pre_qual(x):
    basic = {19, 38}
    incomplete_secondary = {9, 10, 12, 14, 15}
    completed_secondary = {1}
    vocational = {39, 42}
    higher_undergrad = {2, 3, 6, 40}
    postgraduate = {4, 5, 43}
    
    if x in basic or x in incomplete_secondary or x in completed_secondary:
        return 0
    elif x in vocational or x in higher_undergrad or x in postgraduate:
        return 1
    else:
        return 2

def nationality(x):
    lusophone = {1, 21, 22, 24, 25, 26, 41}
    if x in lusophone:
        return 0
    else:
        return 1
    
def qual(x):
    basic = {11, 14, 18, 19, 22, 26, 27, 29, 30, 35, 36, 37, 38}
    secondary = {1, 9, 10, 12}
    higher = {2, 3, 6, 39, 40, 42}
    postgrad = {4, 5, 41, 43, 44}
    
    if x in basic:
        return 0
    elif x in secondary:
        return 1
    elif x in higher:
        return 2
    elif x in postgrad:
        return 3
    else:
        return 4

def moms_job(x):
    student = {0, 90, 99}
    professional = {1, 2, 3, 122, 123, 125, 131, 132, 134}
    admin_service = {4, 5, 141, 143, 144, 151, 152, 153}
    manual = {6, 7, 8, 9, 171, 173, 175, 191, 192, 193, 194}
    military = {10}

    if x in student:
        return 0
    elif x in professional:
        return 1
    elif x in admin_service:
        return 2
    elif x in manual:
        return 3
    elif x in military:
        return 4
    else:
        return 0
    
def dads_job(x):
    student = {0, 90, 99}
    professional = {1, 2, 3, 112, 114, 121, 122, 123, 124, 131, 132, 134, 135}
    admin_service = {4, 5, 141, 143, 144, 151, 152, 153, 154, 195}
    manual = {6, 7, 8, 9, 161, 163, 171, 172, 174, 175, 181, 182, 183, 192, 193, 194}
    military = {10, 101, 102, 103}
    
    if x in student:
        return 0
    elif x in professional:
        return 1
    elif x in admin_service:
        return 2
    elif x in manual:
        return 3
    elif x in military:
        return 4
    else:
        return 0

def load_data():
    _, DATA_PATH, _ = get_project_paths()
    df = pd.read_csv(os.path.join(DATA_PATH, "data.csv"), sep=';')
    return df

def clean_anomalies(df):
    print("   Cleaning anomalies...")
    original_size = len(df)
    
    df = df[(df['Enroll Age'] >= 17) & (df['Enroll Age'] <= 70)]
    df = df[df['Application order'] >= 0]
    df = df[(df['Admission grade'] >= 95) & (df['Admission grade'] <= 190)]
    df = df[(df['Pre Qual (grade)'] >= 95) & (df['Pre Qual (grade)'] <= 190)]

    for sem in ['1st', '2nd']:
        df = df[df[f'{sem} - approved'] <= df[f'{sem} - enrolled']]
        df = df[df[f'{sem} - evaluations'] >= 0]
        df = df[df[f'{sem} - grade'] >= 0]

    df = df[df['Gender'].isin([0, 1])]
    df = df[df['Scholarship'].isin([0, 1])]
    df = df[df['Tuition fees up to date'].isin([0, 1])]
    
    print(f"   Removed {original_size - len(df)} rows with anomalies")
    return df

def group_categorical(df):
    print("   Grouping categorical values...")
    
    df_prep = df.copy()
    useless_cols = ['Debtor', 'Special Needs', 'Unemployment rate', 'Inflation rate', 'GDP']
    useless_cols = [c for c in useless_cols if c in df_prep.columns]
    df_prep = df_prep.drop(columns=useless_cols)

    col_func_map = {
        'Marial Status': lambda x: marial(x),
        'Application mode': lambda x: app_mode(x),
        'Course': lambda x: course(x),
        'Pre Qual': lambda x: pre_qual(x),
        'Nationality': lambda x: nationality(x),
        "Mom's Qual": lambda x: qual(x),
        "Dad's Qual": lambda x: qual(x),
        "Mom's Occupation": lambda x: moms_job(x),
        "Dad's Occupation": lambda x: dads_job(x)
    }

    for col, func in col_func_map.items():
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].apply(func)
    
    return df_prep

def rename_columns(df):
    df.columns = [c.replace('\t', ' ').strip() for c in df.columns]
    df = df.rename(columns={
        'Marital status': 'Marial Status',
        'Daytime/evening attendance': 'Daytime/Evening',
        'Previous qualification': 'Pre Qual',
        'Previous qualification (grade)': 'Pre Qual (grade)',
        'Nacionality': 'Nationality',
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

def visualize_preprocessed(df_prep):
    print("   Creating visualizations...")
    
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    
    cat_candidates = [
        'Marial Status',
        'Application mode',
        'Course',
        'Pre Qual',
        'Nationality',
        "Mom's Qual",
        "Dad's Qual",
        "Mom's Occupation",
        "Dad's Occupation"
    ]
    all_cats = [c for c in cat_candidates if c in df_prep.columns]
    n = len(all_cats)
    cols = 3
    rows = int(np.ceil(n / cols)) if n > 0 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = np.array(axes).flatten()

    legend_handles_labels = None
    for i, col in enumerate(all_cats):
        ct = pd.crosstab(df_prep[col], df_prep['Target'])
        ct = ct.loc[ct.index.sort_values()]
        plot = ct.plot(kind='bar', stacked=True, colormap='tab20', ax=axes[i])
        axes[i].set_title(f'Target count by {col}')
        axes[i].set_ylabel(None)
        axes[i].set_xlabel(None)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45, ha='right')
        if legend_handles_labels is None:
            legend_handles_labels = axes[i].get_legend_handles_labels()
        axes[i].get_legend().remove()

    if legend_handles_labels is not None:
        handles, labels = legend_handles_labels
        fig.legend(handles, labels, title='Target', bbox_to_anchor=(1.05, 1), loc='upper left')

    for j in range(len(all_cats), len(axes)):
        axes[j].axis('off')
    save_fig('target_by_categorical_preprocessed')

def save_preprocessed(df_prep):
    _, DATA_PATH, _ = get_project_paths()
    output_path = os.path.join(DATA_PATH, "preprocessed.csv")
    df_prep.to_csv(output_path, index=False)
    print(f"   Preprocessed data saved to {output_path}")

def run_preprocessing():
    print("=" * 50)
    print("Data Preprocessing")
    print("=" * 50)
    
    print("\n1. Loading data...")
    df = load_data()
    print(f"   Data shape: {df.shape}")
    
    print("\n2. Renaming columns...")
    df = rename_columns(df)
    
    print("\n3. Anomaly cleaning...")
    df = clean_anomalies(df)
    print(f"   Data shape after cleaning: {df.shape}")
    
    print("\n4. Grouping categorical values...")
    df_prep = group_categorical(df)
    
    print("\n5. Basic statistics...")
    print(df_prep.describe())
    
    print("\n6. Saving preprocessed data...")
    save_preprocessed(df_prep)
    
    print("\n7. Visualizing preprocessed data...")
    visualize_preprocessed(df_prep)
    
    print("\n" + "=" * 50)
    print("Preprocessing completed successfully!")
    print("=" * 50)
    return df_prep

if __name__ == "__main__":
    run_preprocessing()