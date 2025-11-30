import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline 

# --- 1. LIVER MODEL ---
def train_liver():
    print("--- (1/5) Training Liver Model ---")
    try:
        df = pd.read_csv('datasets/indian_liver_patient.csv')
        df['Albumin_and_Globulin_Ratio'] = df['Albumin_and_Globulin_Ratio'].fillna(df['Albumin_and_Globulin_Ratio'].median())
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
        df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
        
        features_to_keep = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Albumin']
        X = df[features_to_keep]
        y = df['Dataset']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        pickle.dump(model, open('liver_model_v2.pkl', 'wb'))
        pickle.dump(scaler, open('liver_scaler_v2.pkl', 'wb'))
        print("Liver model saved successfully.\n")
    except Exception as e:
        print(f"Error training liver model: {e}\n")

# --- 2. HEART MODEL ---
def train_heart():
    print("--- (2/5) Training Heart Model ---")
    try:
        df = pd.read_csv('datasets/cardio_train.csv', sep=';')
        df = df.drop('id', axis=1)
        df['age'] = (df['age'] / 365.25).round().astype(int)
        df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['gender'] = df['gender'].map({1: 0, 2: 1}) # 0=Female, 1=Male
        
        df = df[df['ap_lo'] <= df['ap_hi']]
        df = df[(df['ap_hi'] >= 90) & (df['ap_hi'] <= 250)]
        df = df[(df['ap_lo'] >= 60) & (df['ap_lo'] <= 150)]

        features_to_keep = [
            'age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol', 
            'gluc', 'smoke', 'active', 'bmi'
        ]
        X = df[features_to_keep]
        y = df['cardio']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numerical_features = ['age', 'ap_hi', 'ap_lo', 'bmi']
        categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'active']

        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), numerical_features),
                ('passthrough', 'passthrough', categorical_features)
            ],
            remainder='drop'
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        model.fit(X_train_processed, y_train)
        
        pickle.dump(model, open('heart_model_v3.pkl', 'wb'))
        pickle.dump(preprocessor, open('heart_preprocessor_v3.pkl', 'wb'))
        print("Heart model saved successfully.\n")
    except Exception as e:
        print(f"Error training heart model: {e}\n")

# --- 3. KIDNEY MODEL ---
def train_kidney():
    print("--- (3/5) Training Kidney Model ---")
    try:
        df = pd.read_csv('datasets/kidney_disease.csv')
        df.columns = df.columns.str.strip()
        
        features_to_keep = ['age', 'bp', 'sg', 'al', 'su', 'htn', 'pe']
        target = 'classification'
        df_clean = df[features_to_keep + [target]].copy()

        df_clean = df_clean.replace('[?\t]', np.nan, regex=True)
        df_clean['htn'] = df_clean['htn'].map({'yes': 1, 'no': 0})
        df_clean['pe'] = df_clean['pe'].map({'yes': 1, 'no': 0})
        df_clean['classification'] = df_clean['classification'].str.strip()
        df_clean['classification'] = df_clean['classification'].map({'ckd': 1, 'nockd': 0})

        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        df_clean[target] = df_clean[target].astype(int)

        X = df_clean[features_to_keep]
        y = df_clean[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numerical_features = ['age', 'bp']
        categorical_features = ['sg', 'al', 'su', 'htn', 'pe']

        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), numerical_features),
                ('passthrough', 'passthrough', categorical_features)
            ],
            remainder='drop'
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_processed, y_train)
        
        pickle.dump(model, open('kidney_model_v3.pkl', 'wb'))
        pickle.dump(preprocessor, open('kidney_preprocessor_v3.pkl', 'wb'))
        print("Kidney model saved successfully.\n")
    except Exception as e:
        print(f"Error training kidney model: {e}\n")

# --- 4. STROKE MODEL ---
def train_stroke():
    print("--- (4/5) Training Stroke Model (with SMOTE) ---")
    try:
        df = pd.read_csv('datasets/healthcare-dataset-stroke-data.csv')
        df = df.drop('id', axis=1)
        df = df[df['gender'] != 'Other']
        df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
        
        final_features_to_keep = [
            'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        target = 'stroke'
        
        X = df[final_features_to_keep]
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        numerical_features = ['age', 'avg_glucose_level', 'bmi']
        categorical_features = ['gender', 'work_type', 'Residence_type', 'smoking_status']
        binary_features = ['hypertension', 'heart_disease', 'ever_married']

        numerical_transformer = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = ImbPipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features),
                ('bin', 'passthrough', binary_features)
            ],
            remainder='drop'
        )
        
        model_pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        model_pipeline.fit(X_train, y_train)
        
        pickle.dump(model_pipeline, open('stroke_pipeline_v5_SMOTE.pkl', 'wb'))
        print("Stroke pipeline (model+preprocessor+SMOTE) saved successfully.\n")
    except Exception as e:
        print(f"Error training stroke model: {e}\n")

# --- 5. DIABETES MODEL ---
def train_diabetes():
    print("--- (5/5) Training Diabetes Model ---")
    try:
        df = pd.read_csv('datasets/diabetes_data_upload.csv')
        df_clean = df.copy()
        
        yes_no_cols = [
            'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
            'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
            'Irritability', 'delayed healing', 'partial paresis', 
            'muscle stiffness', 'Alopecia', 'Obesity'
        ]
        for col in yes_no_cols:
            df_clean[col] = df_clean[col].map({'Yes': 1, 'No': 0})
        
        df_clean['Gender'] = df_clean['Gender'].map({'Male': 1, 'Female': 0})
        df_clean['class'] = df_clean['class'].map({'Positive': 1, 'Negative': 0})
        df_clean['Thirst_or_Hunger'] = ((df_clean['Polydipsia'] == 1) | (df_clean['Polyphagia'] == 1)).astype(int)

        final_features_to_keep = [
            'Age', 'Gender', 'Polyuria', 'Thirst_or_Hunger', 
            'weakness', 'partial paresis', 'Obesity'
        ]
        target = 'class'
        
        X = df_clean[final_features_to_keep]
        y = df_clean[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        numerical_features = ['Age']
        binary_features = ['Gender', 'Polyuria', 'Thirst_or_Hunger', 'weakness', 'partial paresis', 'Obesity']

        preprocessor = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), numerical_features),
                ('passthrough', 'passthrough', binary_features)
            ],
            remainder='drop'
        )
        
        X_train_processed = preprocessor.fit_transform(X_train)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_processed, y_train)
        
        pickle.dump(model, open('diabetes_model_v3.pkl', 'wb'))
        pickle.dump(preprocessor, open('diabetes_preprocessor_v3.pkl', 'wb'))
        print("Diabetes model saved successfully.\n")
    except Exception as e:
        print(f"Error training diabetes model: {e}\n")

# --- Main function to run all training ---
if __name__ == '__main__':
    train_liver()
    train_heart()
    train_kidney()
    train_stroke()
    train_diabetes()
    print("--- ALL MODELS HAVE BEEN RE-TRAINED AND SAVED ---")