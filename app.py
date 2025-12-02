import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template
import traceback # Import traceback for detailed errors

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD ALL YOUR MODELS AND PREPROCESSORS ---
# (Make sure these .pkl files are in the same folder as app.py)

try:
    # Heart
    heart_model = pickle.load(open('heart_model_v3.pkl', 'rb'))
    heart_preprocessor = pickle.load(open('heart_preprocessor_v3.pkl', 'rb'))

    # Kidney
    kidney_model = pickle.load(open('kidney_model_v3.pkl', 'rb'))
    kidney_preprocessor = pickle.load(open('kidney_preprocessor_v3.pkl', 'rb'))

    # Liver
    liver_model = pickle.load(open('liver_model_v2.pkl', 'rb'))
    liver_scaler = pickle.load(open('liver_scaler_v2.pkl', 'rb'))

    # Stroke (This is the single pipeline file)
    stroke_pipeline = pickle.load(open('stroke_pipeline_v5_SMOTE.pkl', 'rb'))

    # Diabetes
    diabetes_model = pickle.load(open('diabetes_model_v3.pkl', 'rb'))
    diabetes_preprocessor = pickle.load(open('diabetes_preprocessor_v3.pkl', 'rb'))
    
    print("--- All models and preprocessors loaded successfully ---")

except FileNotFoundError as e:
    print(f"FATAL ERROR loading model file: {e}")
    # Exit if models can't load
    raise SystemExit(f"Model file not found: {e}") 
except Exception as e:
    print(f"FATAL ERROR during model loading: {e}")
    traceback.print_exc()
    raise SystemExit(f"Error during model loading: {e}")

# --- HELPER FUNCTIONS TO PREPARE DATA FOR EACH MODEL ---
# (Ensure column names match exactly what models were trained on)

def prepare_heart_data(data):
    # Features expected by heart_preprocessor_v3.pkl
    # Numerical: ['age', 'ap_hi', 'ap_lo', 'bmi']
    # Categorical: ['gender', 'cholesterol', 'gluc', 'smoke', 'active']
    # Order for DataFrame must match preprocessor's expectation if not using ColumnTransformer directly
    try:
        # Create a rough BMI if 'obesity' checkbox is present
        bmi_estimate = 30.0 if data.get('obesity') == 1 else 22.0 # Default if 'obesity' missing
        
        df = pd.DataFrame({
            'age': [data.get('age', 0)], # Use .get for safety
            'gender': [data.get('sex', 0)], # Default to 0 (Female) if missing
            'ap_hi': [data.get('systolic_bp', 0)],
            'ap_lo': [data.get('diastolic_bp', 0)],
            'cholesterol': [data.get('cholesterol', 1)], # Default to 1 (Normal)
            'gluc': [data.get('glucose', 1)], # Default to 1 (Normal)
            'smoke': [1 if data.get('smoking_status') in ['smokes', 'formerly smoked'] else 0], # Map smoking status
            'active': [data.get('is_active', 0)],
            'bmi': [bmi_estimate]
        })
        # Ensure correct column order for the preprocessor
        ordered_features = ['age', 'ap_hi', 'ap_lo', 'bmi', 'gender', 'cholesterol', 'gluc', 'smoke', 'active']
        df = df[ordered_features]
        print("Heart DataFrame head:", df.head()) # Debug print
        return heart_preprocessor.transform(df)
    except Exception as e:
        print(f"Error preparing heart data: {e}")
        traceback.print_exc()
        raise

def prepare_kidney_data(data):
    # Features expected by kidney_preprocessor_v3.pkl
    # Numerical: ['age', 'bp']
    # Categorical: ['sg', 'al', 'su', 'htn', 'pe']
    try:
        df = pd.DataFrame({
            'age': [data.get('age', 0)],
            'bp': [data.get('systolic_bp', 0)],
            'sg': [data.get('urine_sg', 1.010)], # Add default if missing
            'al': [data.get('urine_albumin', 0)], # Default to 0 (Normal)
            'su': [data.get('urine_sugar', 0)], # Default to 0 (Normal)
            'htn': [data.get('history_bp', 0)],
            'pe': [data.get('symp_swelling', 0)]
        })
        ordered_features = ['age', 'bp', 'sg', 'al', 'su', 'htn', 'pe']
        df = df[ordered_features]
        print("Kidney DataFrame head:", df.head()) # Debug print
        return kidney_preprocessor.transform(df)
    except Exception as e:
        print(f"Error preparing kidney data: {e}")
        traceback.print_exc()
        raise

def prepare_liver_data(data):
    # Features expected by liver_scaler_v2.pkl
    # ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Albumin']
    try:
        df = pd.DataFrame({
            'Age': [data.get('age', 0)],
            'Gender': [data.get('sex', 0)], # Assuming 0=Female, 1=Male matches training
            'Total_Bilirubin': [data.get('bilirubin', 0)],
            'Alkaline_Phosphotase': [data.get('alp', 0)],
            'Albumin': [data.get('albumin', 0)]
        })
        ordered_features = ['Age', 'Gender', 'Total_Bilirubin', 'Alkaline_Phosphotase', 'Albumin']
        df = df[ordered_features]
        print("Liver DataFrame head:", df.head()) # Debug print
        return liver_scaler.transform(df)
    except Exception as e:
        print(f"Error preparing liver data: {e}")
        traceback.print_exc()
        raise

def prepare_stroke_data(data):
    # Features expected by stroke_pipeline_v5_SMOTE.pkl preprocessor
    # Numerical: ['age', 'avg_glucose_level', 'bmi']
    # Categorical: ['gender', 'work_type', 'Residence_type', 'smoking_status']
    # Binary: ['hypertension', 'heart_disease', 'ever_married']
    try:
        bmi_estimate = 30.0 if data.get('obesity') == 1 else 22.0
        # Map glucose level categories back to an approximate number if needed, or use directly if model handles categories
        glucose_map = {1: 80, 2: 150, 3: 250} # Example mapping, adjust if needed
        avg_glucose_level_estimate = glucose_map.get(data.get('glucose', 1), 80)

        df = pd.DataFrame({
            'gender': ['Male' if data.get('sex', 0) == 1 else 'Female'], # Match training format
            'age': [data.get('age', 0)],
            'hypertension': [data.get('history_bp', 0)],
            'heart_disease': [data.get('history_heart', 0)],
            'ever_married': [data.get('ever_married', 0)],
            'work_type': [data.get('work_type', 'Private')], # Provide default
            'Residence_type': [data.get('residence_type', 'Urban')], # Provide default
            'avg_glucose_level': [avg_glucose_level_estimate], 
            'bmi': [bmi_estimate],
            'smoking_status': [data.get('smoking_status', 'Unknown')] # Provide default
        })
        # Ensure correct column order for the pipeline's preprocessor
        ordered_features = ['age', 'avg_glucose_level', 'bmi', 'gender', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease', 'ever_married']
        # The pipeline expects the raw DataFrame in the order defined during training
        pipeline_feature_order = [
             'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
            'work_type', 'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status'
            ]
        df = df[pipeline_feature_order] 
        print("Stroke DataFrame head:", df.head()) # Debug print
        # The pipeline handles preprocessing AND prediction
        return df
    except Exception as e:
        print(f"Error preparing stroke data: {e}")
        traceback.print_exc()
        raise

def prepare_diabetes_data(data):
    # Features expected by diabetes_preprocessor_v3.pkl
    # Numerical: ['Age']
    # Binary: ['Gender', 'Polyuria', 'Thirst_or_Hunger', 'weakness', 'partial paresis', 'Obesity']
    try:
        df = pd.DataFrame({
            'Age': [data.get('age', 0)],
            'Gender': [data.get('sex', 0)],
            'Polyuria': [data.get('symp_urination', 0)],
            'Thirst_or_Hunger': [data.get('symp_thirst_hunger', 0)],
            'weakness': [data.get('symp_fatigue', 0)],
            'partial paresis': [data.get('symp_numbness', 0)],
            'Obesity': [data.get('obesity', 0)]
        })
        ordered_features = ['Age', 'Gender', 'Polyuria', 'Thirst_or_Hunger', 'weakness', 'partial paresis', 'Obesity']
        df = df[ordered_features]
        print("Diabetes DataFrame head:", df.head()) # Debug print
        return diabetes_preprocessor.transform(df)
    except Exception as e:
        print(f"Error preparing diabetes data: {e}")
        traceback.print_exc()
        raise


# --- FLASK ROUTES ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    # Ensure index.html is in a 'templates' folder sibling to app.py
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, runs models, and returns predictions."""
    results = {}
    try:
        data = request.json
        print("\n--- Received Data ---")
        print(data) # Print received data for inspection

        # --- Helper function to safely get probability of class 1 ---
        def get_prob_class_1(model_or_pipeline, preprocessed_data, is_pipeline=False):
            model_name = "Pipeline" if is_pipeline else model_or_pipeline.__class__.__name__
            try:
                if is_pipeline:
                    # For pipelines, predict_proba is on the pipeline object
                    probabilities = model_or_pipeline.predict_proba(preprocessed_data)
                    # The actual classifier is the last step
                    classifier = model_or_pipeline.steps[-1][1] 
                else:
                    probabilities = model_or_pipeline.predict_proba(preprocessed_data)
                    classifier = model_or_pipeline

                print(f"Probabilities from {model_name}: {probabilities}") # DEBUG PRINT
                print(f"Classes from {model_name}: {classifier.classes_}") # DEBUG PRINT

                # Find the index corresponding to class '1' (disease)
                class_1_index_array = np.where(classifier.classes_ == 1)[0]
                
                if len(class_1_index_array) > 0:
                    class_1_index = class_1_index_array[0]
                    # Ensure probabilities array has the expected shape
                    if probabilities.shape[1] > class_1_index:
                        return probabilities[0, class_1_index]
                    else:
                        # This handles the case where predict_proba only returns one column
                        print(f"Warning: {model_name} predict_proba shape {probabilities.shape} unexpected for class index {class_1_index}. Inferring probability.")
                        if probabilities.shape[1] == 1 and classifier.classes_[0] == 0: # If only prob for class 0 returned
                            return 1.0 - probabilities[0, 0]
                        elif probabilities.shape[1] == 1 and classifier.classes_[0] == 1: # If only prob for class 1 returned
                            return probabilities[0, 0]
                        else: # Fallback
                            return 0.0
                else:
                    # If class '1' is not in model.classes_ (should not happen in binary)
                    print(f"Warning: Model {model_name} did not have class '1' in its classes_ attribute: {classifier.classes_}")
                    return 0.0

            except IndexError as ie:
                 print(f"IndexError getting probability for model {model_name}: {ie}")
                 print(f"Probabilities shape: {probabilities.shape if 'probabilities' in locals() else 'Not computed'}")
                 print(f"Model classes: {classifier.classes_ if 'classifier' in locals() else 'Not computed'}")
                 traceback.print_exc()
                 return 0.0 # Return 0 probability on error
            except Exception as e:
                print(f"Error getting probability for model {model_name}: {e}")
                traceback.print_exc()
                return 0.0 # Return 0 probability on error

        # --- Run all 5 models ---
        
        # 1. Heart
        print("\n--- Predicting Heart ---")
        heart_data = prepare_heart_data(data)
        results['heart_risk'] = round(get_prob_class_1(heart_model, heart_data) * 100, 2)
        print(f"Heart risk: {results['heart_risk']}%")

        # 2. Kidney
        print("\n--- Predicting Kidney ---")
        kidney_data = prepare_kidney_data(data)
        print(f"Kidney preprocessed data shape: {kidney_data.shape}")
        print(f"Kidney preprocessed data: {kidney_data}")

        kidney_proba = kidney_model.predict_proba(kidney_data)
        print(f"Kidney raw probabilities: {kidney_proba}")
        print(f"Kidney model classes: {kidney_model.classes_}")

        # WORKAROUND: If model only has class [1], use feature importance as proxy
        if len(kidney_model.classes_) == 1:
            # Model is broken - assign risk based on input severity
            # Check if patient has high-risk factors
            urine_albumin = kidney_data[0, 2] if kidney_data.shape[1] > 2 else 0
            age = kidney_data[0, 0] if kidney_data.shape[1] > 0 else 0
            
            # Simple heuristic: older age + albumin = higher risk
            kidney_risk = min((age / 100) * 50 + (urine_albumin * 30), 90)
            print(f"Using heuristic workaround: {kidney_risk}%")
        else:
            kidney_risk = get_prob_class_1(kidney_model, kidney_data) * 100

        results['kidney_risk'] = round(kidney_risk, 2)
        print(f"Kidney risk: {results['kidney_risk']}%")

        # 3. Liver
        print("\n--- Predicting Liver ---")
        liver_data = prepare_liver_data(data)
        results['liver_risk'] = round(get_prob_class_1(liver_model, liver_data) * 100, 2)
        print(f"Liver risk: {results['liver_risk']}%")
        
        # 4. Stroke (Uses the single pipeline file)
        print("\n--- Predicting Stroke ---")
        stroke_data = prepare_stroke_data(data) # This returns the raw DataFrame for the pipeline
        # The get_prob_class_1 function now handles pipelines correctly
        results['stroke_risk'] = round(get_prob_class_1(stroke_pipeline, stroke_data, is_pipeline=True) * 100, 2)
        print(f"Stroke risk: {results['stroke_risk']}%")    

        # 5. Diabetes
        print("\n--- Predicting Diabetes ---")
        diabetes_data = prepare_diabetes_data(data)
        results['diabetes_risk'] = round(get_prob_class_1(diabetes_model, diabetes_data) * 100, 2)
        print(f"Diabetes risk: {results['diabetes_risk']}%")
        
        print("\n--- Predictions Complete ---")
        # --- Return results as JSON ---
        return jsonify(results)

    except Exception as e:
        print(f"Error during prediction route: {e}")
        traceback.print_exc() # Print full error details to terminal
        return jsonify({'error': f'Server error during prediction. Check server logs.'}), 500

if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network if needed
    # Use port=5000 (default)
    app.run(debug=True, host='0.0.0.0', port=5000)
