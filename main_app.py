# main_app.py - Unified AutoML Backend
from flask import Flask, request, jsonify
from flask_cors import CORS
import h2o
from h2o.estimators import H2ORandomForestEstimator, H2OGradientBoostingEstimator, H2OGeneralizedLinearEstimator, H2OKMeansEstimator
from h2o.automl import H2OAutoML
import pandas as pd
import pickle
import os
import tempfile
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global dictionary to store models
models = {}

def get_model_choice(choice, task_type='classification'):
    """Get model based on choice and task type"""
    if task_type == 'classification':
        models_dict = {
            '1': H2ORandomForestEstimator(ntrees=100, max_depth=20, seed=1),
            '2': H2OGradientBoostingEstimator(ntrees=100, max_depth=20, seed=1),
            '3': H2OGeneralizedLinearEstimator(family='binomial'),
            '4': H2OAutoML(max_models=10, max_runtime_secs=3600, seed=1)
        }
    elif task_type == 'regression':
        models_dict = {
            '1': H2ORandomForestEstimator(ntrees=100, max_depth=20, seed=1),
            '2': H2OGradientBoostingEstimator(ntrees=100, max_depth=20, seed=1),
            '3': H2OGeneralizedLinearEstimator(family='gaussian'),
            '4': H2OAutoML(max_models=10, max_runtime_secs=3600, seed=1)
        }
    elif task_type == 'clustering':
        models_dict = {
            '1': H2OKMeansEstimator(k=3, seed=1),
            '2': H2OAutoML(max_models=10, max_runtime_secs=3600, seed=1)
        }
    else:
        models_dict = {
            '4': H2OAutoML(max_models=10, max_runtime_secs=3600, seed=1)
        }
    
    return models_dict.get(choice, H2OAutoML(max_models=10, max_runtime_secs=3600, seed=1))

def load_data_from_content(content, file_type, filename):
    """Load data from file content"""
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    temp_filename = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
    
    try:
        # Write content to temporary file
        with open(temp_filename, 'wb') as f:
            f.write(content)
        
        # Load based on file type
        if file_type.lower() == 'csv':
            df = pd.read_csv(temp_filename)
        elif file_type.lower() in ['tabular', 'tsv']:
            df = pd.read_table(temp_filename)
        elif file_type.lower() in ['excel', 'xlsx', 'xls']:
            df = pd.read_excel(temp_filename)
        else:
            raise ValueError("Unsupported file type. Please use 'csv', 'tabular', or 'excel'.")
        
        return df
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def preprocess_data(df, target=None, task_type='classification'):
    """Preprocess data for ML"""
    # Remove date columns
    for col in df.columns:
        try:
            pd.to_datetime(df[col], format='%d-%m-%Y', errors='raise')
            df.drop(columns=[col], inplace=True)
        except:
            continue
    
    # Handle missing values
    if target and target in df.columns:
        df = df.dropna(subset=[target])
    
    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    # Encode categorical variables
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes
    
    # For classification, ensure target is categorical if it has few unique values
    if target and task_type == 'classification' and target in df.columns:
        if df[target].nunique() <= 10:
            df[target] = df[target].astype('category')
    
    return df

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "AutoML Dashboard Backend API",
        "version": "1.0.0",
        "endpoints": {
            "/classification/train": "Train classification model",
            "/classification/predict": "Make predictions with classification model",
            "/regression/train": "Train regression model", 
            "/regression/predict": "Make predictions with regression model",
            "/clustering/train": "Train clustering model",
            "/clustering/predict": "Make predictions with clustering model",
            "/health": "Health check"
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "h2o_available": True})

@app.route('/classification/train', methods=['POST'])
def train_classification():
    try:
        # Get file from request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        target = request.form.get('target')
        model_choice = request.form.get('model_choice', '4')
        
        if not target:
            return jsonify({"error": "Target column not specified"}), 400
        
        # Load and preprocess data
        file_content = file.read()
        file_type = file.filename.split('.')[-1]
        df = load_data_from_content(file_content, file_type, file.filename)
        df = preprocess_data(df, target, 'classification')
        
        # Initialize H2O
        h2o.init()
        
        # Convert to H2O frame
        data = h2o.H2OFrame(df)
        train_data, test_data = data.split_frame(ratios=[0.8], seed=1)
        features = [col for col in train_data.columns if col != target]
        
        # Train model
        model = get_model_choice(model_choice, 'classification')
        
        if isinstance(model, H2OAutoML):
            model.train(x=features, y=target, training_frame=train_data)
            # For AutoML, get the leader model
            leader_model = model.leader
            model_path = h2o.save_model(model=leader_model, path="./models", force=True)
        else:
            model.train(x=features, y=target, training_frame=train_data)
            model_path = h2o.save_model(model=model, path="./models", force=True)
        
        # Get predictions and calculate accuracy
        if isinstance(model, H2OAutoML):
            predictions = model.leader.predict(test_data)
        else:
            predictions = model.predict(test_data)
        predictions_df = predictions.as_data_frame()
        test_data_df = test_data.as_data_frame()
        
        # Calculate accuracy
        valid_targets = train_data[target].as_data_frame()[target].unique()
        predictions_df['predict'] = predictions_df['predict'].apply(
            lambda x: x if x in valid_targets else valid_targets[0]
        )
        correct_predictions = (predictions_df['predict'] == test_data_df[target]).sum()
        accuracy = correct_predictions / len(test_data_df)
        
        # Save model
        model_id = str(uuid.uuid4())
        models[model_id] = {
            'path': model_path,
            'type': 'classification',
            'features': features,
            'target': target
        }
        
        h2o.shutdown(prompt=False)
        
        return jsonify({
            "model_id": model_id,
            "model_path": model_path,
            "accuracy": float(accuracy * 100),
            "features": features,
            "target": target
        })
        
    except Exception as e:
        h2o.shutdown(prompt=False)
        return jsonify({"error": str(e)}), 500

@app.route('/regression/train', methods=['POST'])
def train_regression():
    try:
        # Get file from request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        target = request.form.get('target')
        model_choice = request.form.get('model_choice', '4')
        
        if not target:
            return jsonify({"error": "Target column not specified"}), 400
        
        # Load and preprocess data
        file_content = file.read()
        file_type = file.filename.split('.')[-1]
        df = load_data_from_content(file_content, file_type, file.filename)
        df = preprocess_data(df, target, 'regression')
        
        # Initialize H2O
        h2o.init()
        
        # Convert to H2O frame
        data = h2o.H2OFrame(df)
        train_data, test_data = data.split_frame(ratios=[0.8], seed=1)
        features = [col for col in train_data.columns if col != target]
        
        # Train model
        model = get_model_choice(model_choice, 'regression')
        
        if isinstance(model, H2OAutoML):
            model.train(x=features, y=target, training_frame=train_data)
            # For AutoML, get the leader model
            leader_model = model.leader
            model_path = h2o.save_model(model=leader_model, path="./models", force=True)
        else:
            model.train(x=features, y=target, training_frame=train_data)
            model_path = h2o.save_model(model=model, path="./models", force=True)
        
        # Get predictions and calculate metrics
        if isinstance(model, H2OAutoML):
            predictions = model.leader.predict(test_data)
        else:
            predictions = model.predict(test_data)
        predictions_df = predictions.as_data_frame()
        test_data_df = test_data.as_data_frame()
        
        # Calculate RMSE
        rmse = ((predictions_df['predict'] - test_data_df[target]) ** 2).mean() ** 0.5
        
        # Save model
        model_id = str(uuid.uuid4())
        models[model_id] = {
            'path': model_path,
            'type': 'regression',
            'features': features,
            'target': target
        }
        
        h2o.shutdown(prompt=False)
        
        return jsonify({
            "model_id": model_id,
            "model_path": model_path,
            "rmse": float(rmse),
            "features": features,
            "target": target
        })
        
    except Exception as e:
        h2o.shutdown(prompt=False)
        return jsonify({"error": str(e)}), 500

@app.route('/clustering/train', methods=['POST'])
def train_clustering():
    try:
        # Get file from request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        model_choice = request.form.get('model_choice', '1')
        k_clusters = int(request.form.get('k_clusters', '3'))
        
        # Load and preprocess data
        file_content = file.read()
        file_type = file.filename.split('.')[-1]
        df = load_data_from_content(file_content, file_type, file.filename)
        df = preprocess_data(df, None, 'clustering')
        
        # Initialize H2O
        h2o.init()
        
        # Convert to H2O frame
        data = h2o.H2OFrame(df)
        features = list(data.columns)
        
        # Train model
        if model_choice == '1':
            model = H2OKMeansEstimator(k=k_clusters, seed=1)
        else:
            model = H2OAutoML(max_models=10, max_runtime_secs=3600, seed=1)
        
        model.train(x=features, training_frame=data)
        
        # Get cluster assignments
        predictions = model.predict(data)
        predictions_df = predictions.as_data_frame()
        
        # Save model
        model_id = str(uuid.uuid4())
        model_path = h2o.save_model(model=model, path="./models", force=True)
        models[model_id] = {
            'path': model_path,
            'type': 'clustering',
            'features': features,
            'k_clusters': k_clusters
        }
        
        h2o.shutdown(prompt=False)
        
        return jsonify({
            "model_id": model_id,
            "model_path": model_path,
            "features": features,
            "k_clusters": k_clusters,
            "cluster_assignments": predictions_df.to_dict(orient="records")[:10]  # Return first 10 for preview
        })
        
    except Exception as e:
        h2o.shutdown(prompt=False)
        return jsonify({"error": str(e)}), 500

@app.route('/<task_type>/predict', methods=['POST'])
def predict(task_type):
    try:
        # Get file and model ID from request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        model_id = request.form.get('model_id')
        
        if not model_id or model_id not in models:
            return jsonify({"error": "Invalid model ID"}), 400
        
        model_info = models[model_id]
        
        # Load and preprocess data
        file_content = file.read()
        file_type = file.filename.split('.')[-1]
        df = load_data_from_content(file_content, file_type, file.filename)
        df = preprocess_data(df, None, task_type)
        
        # Initialize H2O and load model
        h2o.init()
        model = h2o.load_model(model_info['path'])
        
        # Make predictions
        data = h2o.H2OFrame(df)
        predictions = model.predict(data)
        predictions_df = predictions.as_data_frame()
        
        h2o.shutdown(prompt=False)
        
        return jsonify({
            "predictions": predictions_df.to_dict(orient="records"),
            "model_id": model_id,
            "model_type": model_info['type']
        })
        
    except Exception as e:
        h2o.shutdown(prompt=False)
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
