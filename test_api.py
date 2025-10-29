#!/usr/bin/env python3
"""
Test script for the AutoML Dashboard Backend API
This script demonstrates how to use the unified API for different ML tasks.
"""

import requests
import pandas as pd
import numpy as np
import io
import time

# API base URL
BASE_URL = "http://127.0.0.1:5000"

def create_sample_csv():
    """Create a sample CSV file for testing"""
    # Create sample classification data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples), 
        'feature3': np.random.uniform(0, 10, n_samples),
        'feature4': np.random.randint(1, 5, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    }
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

def create_sample_regression_csv():
    """Create a sample CSV file for regression testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Create some features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(0, 1, n_samples)
    feature3 = np.random.uniform(0, 10, n_samples)
    
    # Create target with some relationship to features
    target = 2 * feature1 + 3 * feature2 + 0.5 * feature3 + np.random.normal(0, 0.5, n_samples)
    
    data = {
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'target': target
    }
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    return csv_buffer.getvalue().encode('utf-8')

def test_api_health():
    """Test API health check"""
    print("🔍 Testing API health...")
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("✅ API is healthy!")
        print(f"Response: {response.json()}")
    else:
        print(f"❌ API health check failed: {response.status_code}")
    print()

def test_classification():
    """Test classification training"""
    print("🤖 Testing Classification Training...")
    
    # Create sample data
    csv_data = create_sample_csv()
    
    # Prepare files and data
    files = {'file': ('test_data.csv', csv_data, 'text/csv')}
    data = {
        'target': 'target',
        'model_choice': '4'  # AutoML
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/classification/train", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Classification model trained successfully!")
        print(f"Model ID: {result['model_id']}")
        print(f"Accuracy: {result['accuracy']:.2f}%")
        print(f"Features: {result['features']}")
        return result['model_id']
    else:
        print(f"❌ Classification training failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None
    print()

def test_regression():
    """Test regression training"""
    print("📈 Testing Regression Training...")
    
    # Create sample data
    csv_data = create_sample_regression_csv()
    
    # Prepare files and data
    files = {'file': ('test_regression_data.csv', csv_data, 'text/csv')}
    data = {
        'target': 'target',
        'model_choice': '1'  # Random Forest
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/regression/train", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Regression model trained successfully!")
        print(f"Model ID: {result['model_id']}")
        print(f"RMSE: {result['rmse']:.4f}")
        print(f"Features: {result['features']}")
        return result['model_id']
    else:
        print(f"❌ Regression training failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None
    print()

def test_clustering():
    """Test clustering"""
    print("🎯 Testing Clustering...")
    
    # Create sample data (without target)
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples),
        'feature3': np.random.uniform(0, 10, n_samples),
    }
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue().encode('utf-8')
    
    # Prepare files and data
    files = {'file': ('test_clustering_data.csv', csv_data, 'text/csv')}
    data = {
        'model_choice': '1',  # K-Means
        'k_clusters': '3'
    }
    
    # Send request
    response = requests.post(f"{BASE_URL}/clustering/train", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print("✅ Clustering model trained successfully!")
        print(f"Model ID: {result['model_id']}")
        print(f"Number of clusters: {result['k_clusters']}")
        print(f"Features: {result['features']}")
        print(f"Sample cluster assignments: {len(result['cluster_assignments'])} samples")
        return result['model_id']
    else:
        print(f"❌ Clustering training failed: {response.status_code}")
        print(f"Error: {response.text}")
        return None
    print()

def test_prediction(model_id, task_type):
    """Test prediction with a trained model"""
    print(f"🔮 Testing {task_type.title()} Prediction...")
    
    if task_type == 'classification':
        csv_data = create_sample_csv()
    elif task_type == 'regression':
        csv_data = create_sample_regression_csv()
    else:  # clustering
        np.random.seed(123)  # Different seed for test data
        n_samples = 100
        data = {
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.uniform(0, 10, n_samples),
        }
        df = pd.DataFrame(data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')
    
    # Prepare files and data
    files = {'file': ('test_prediction_data.csv', csv_data, 'text/csv')}
    data = {'model_id': model_id}
    
    # Send request
    response = requests.post(f"{BASE_URL}/{task_type}/predict", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ {task_type.title()} prediction successful!")
        print(f"Number of predictions: {len(result['predictions'])}")
        print(f"Model type: {result['model_type']}")
        print(f"Sample predictions (first 5): {result['predictions'][:5]}")
    else:
        print(f"❌ {task_type.title()} prediction failed: {response.status_code}")
        print(f"Error: {response.text}")
    print()

def main():
    """Main test function"""
    print("🚀 Starting AutoML Dashboard Backend API Tests")
    print("=" * 50)
    
    # Test API health
    test_api_health()
    
    # Test classification
    classification_model_id = test_classification()
    
    # Test regression
    regression_model_id = test_regression()
    
    # Test clustering
    clustering_model_id = test_clustering()
    
    # Test predictions if models were trained successfully
    if classification_model_id:
        test_prediction(classification_model_id, 'classification')
    
    if regression_model_id:
        test_prediction(regression_model_id, 'regression')
    
    if clustering_model_id:
        test_prediction(clustering_model_id, 'clustering')
    
    print("🎉 All tests completed!")

if __name__ == "__main__":
    main()
