# AutoML Dashboard Backend

A unified Flask-based backend API for automated machine learning tasks including classification, regression, and clustering using H2O.ai.

## Features

- **Classification**: Train and predict with classification models
- **Regression**: Train and predict with regression models  
- **Clustering**: Train and predict with clustering models
- **Multiple Algorithms**: Random Forest, Gradient Boosting, GLM, and AutoML
- **File Support**: CSV, Excel, and tabular files
- **REST API**: Easy-to-use REST endpoints
- **CORS Enabled**: Ready for frontend integration

## Installation

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/Mac
# or
.venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Server

```bash
python main_app.py
```

The server will start on `http://localhost:5000`

## API Endpoints

### Health Check
- **GET** `/health` - Check API health and H2O availability
- **GET** `/` - Get API information and available endpoints

### Classification
- **POST** `/classification/train` - Train a classification model
- **POST** `/classification/predict` - Make predictions with a trained classification model

### Regression  
- **POST** `/regression/train` - Train a regression model
- **POST** `/regression/predict` - Make predictions with a trained regression model

### Clustering
- **POST** `/clustering/train` - Train a clustering model
- **POST** `/clustering/predict` - Make predictions with a trained clustering model

## Usage Examples

### Training a Classification Model

```bash
curl -X POST http://localhost:5000/classification/train \
  -F "file=@your_data.csv" \
  -F "target=target_column" \
  -F "model_choice=4"
```

**Parameters:**
- `file`: CSV/Excel file with your data
- `target`: Name of the target column
- `model_choice`: 1=Random Forest, 2=GBM, 3=GLM, 4=AutoML (default)

**Response:**
```json
{
  "model_id": "uuid-string",
  "model_path": "path/to/saved/model",
  "accuracy": 85.5,
  "features": ["feature1", "feature2", "feature3"],
  "target": "target_column"
}
```

### Making Predictions

```bash
curl -X POST http://localhost:5000/classification/predict \
  -F "file=@new_data.csv" \
  -F "model_id=your-model-id"
```

**Response:**
```json
{
  "predictions": [
    {"predict": 1, "p0": 0.3, "p1": 0.7},
    {"predict": 0, "p0": 0.8, "p1": 0.2}
  ],
  "model_id": "your-model-id",
  "model_type": "classification"
}
```

### Training a Regression Model

```bash
curl -X POST http://localhost:5000/regression/train \
  -F "file=@your_data.csv" \
  -F "target=price" \
  -F "model_choice=1"
```

### Training a Clustering Model

```bash
curl -X POST http://localhost:5000/clustering/train \
  -F "file=@your_data.csv" \
  -F "model_choice=1" \
  -F "k_clusters=3"
```

## Model Choices

### Classification & Regression
- `1`: Random Forest
- `2`: Gradient Boosting Machine (GBM)
- `3`: Generalized Linear Model (GLM)
- `4`: AutoML (default - automatically selects best model)

### Clustering
- `1`: K-Means Clustering
- `2`: AutoML

## Testing

Run the test script to verify all endpoints:

```bash
python test_api.py
```

This will test:
- API health check
- Classification training and prediction
- Regression training and prediction  
- Clustering training and prediction

## File Formats

Supported file formats:
- **CSV**: `.csv` files
- **Excel**: `.xlsx`, `.xls` files
- **Tabular**: `.tsv`, `.txt` tab-separated files

## Data Preprocessing

The API automatically handles:
- Missing value imputation
- Categorical variable encoding
- Date column removal
- Train/test splitting (80/20)

## Dependencies

- Flask 3.1.2
- H2O 3.46.0.8
- Pandas 2.3.3
- NumPy 2.3.4
- Flask-CORS 6.0.1
- scikit-learn 1.7.2

## Error Handling

The API returns appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request (missing parameters, invalid file)
- `500`: Internal Server Error

Error responses include descriptive messages:
```json
{
  "error": "Target column not specified"
}
```

## Model Storage

Trained models are automatically saved in the `./models` directory and tracked with unique IDs for later prediction use.

## CORS

CORS is enabled for all routes, allowing frontend applications to make requests from different origins.

## Development

For development, the server runs with debug mode enabled and auto-reloading on file changes.

## Production Deployment

For production deployment, consider using a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 main_app:app
```
