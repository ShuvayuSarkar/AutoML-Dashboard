#!/usr/bin/env python3
import requests
import json

def test_classification():
    print("Testing Classification Training...")
    
    # Use the sample data file
    with open('/home/delicatemedic/Documents/Web/AutoML-Dashboard/sample_data.csv', 'rb') as f:
        files = {'file': ('sample_data.csv', f, 'text/csv')}
        data = {
            'target': 'target',
            'model_choice': '1'  # Random Forest
        }
        
        response = requests.post('http://127.0.0.1:5000/classification/train', 
                               files=files, data=data, timeout=60)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Model ID: {result['model_id']}")
            print(f"Accuracy: {result['accuracy']:.2f}%")
            print(f"Features: {result['features']}")
            print(f"Target: {result['target']}")
            return result['model_id']
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            return None

def test_regression():
    print("\nTesting Regression Training...")
    
    # Create regression data
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    n = 50
    x1 = np.random.normal(0, 1, n)
    x2 = np.random.normal(0, 1, n)
    y = 2*x1 + 3*x2 + np.random.normal(0, 0.1, n)
    
    df = pd.DataFrame({'feature1': x1, 'feature2': x2, 'target': y})
    df.to_csv('/tmp/regression_data.csv', index=False)
    
    with open('/tmp/regression_data.csv', 'rb') as f:
        files = {'file': ('regression_data.csv', f, 'text/csv')}
        data = {
            'target': 'target',
            'model_choice': '1'  # Random Forest
        }
        
        response = requests.post('http://127.0.0.1:5000/regression/train', 
                               files=files, data=data, timeout=60)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ SUCCESS!")
            print(f"Model ID: {result['model_id']}")
            print(f"RMSE: {result['rmse']:.4f}")
            print(f"Features: {result['features']}")
            return result['model_id']
        else:
            print("❌ FAILED!")
            print(f"Error: {response.text}")
            return None

def test_health():
    print("Testing Health Endpoint...")
    response = requests.get('http://127.0.0.1:5000/health')
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ API is healthy!")
        print(f"Response: {response.json()}")
    else:
        print("❌ Health check failed!")

def main():
    print("🚀 AutoML Dashboard Backend Tests")
    print("=" * 40)
    
    # Test health
    test_health()
    
    # Test classification
    class_model_id = test_classification()
    
    # Test regression  
    reg_model_id = test_regression()
    
    print("\n" + "=" * 40)
    print("📊 RESULTS SUMMARY")
    print(f"Health Check: ✅")
    print(f"Classification: {'✅' if class_model_id else '❌'}")
    print(f"Regression: {'✅' if reg_model_id else '❌'}")
    
    if class_model_id and reg_model_id:
        print("\n🎉 All tests passed! Backend is fully functional!")
    else:
        print("\n⚠️  Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main()
