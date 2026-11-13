# SentinelFlow Machine Learning Module (`sentinelflow_ml`)

This module contains the independent machine learning classification component for the industrial SCADA-based water treatment monitoring project, **SentinelFlow**.

## Project Architecture
```text
SentinelFlow_ML/
├── dataset/
│   └── scada_synthetic_dataset.csv     # 10,000 generated SCADA records
├── models/
│   └── sentinelflow_model.joblib       # Serialized sklearn pipeline & metadata
├── results/
│   ├── evaluation_metrics.json         # Accuracy, Precision, Recall, F1
│   ├── confusion_matrix.png            # Confusion matrix plot
│   └── feature_importance.png          # Feature importance plot
├── sentinelflow_ml/                    # Reusable Python package for FastAPI
│   ├── __init__.py
│   ├── schemas.py                      # Pydantic input/output validation
│   └── predictor.py                    # Singleton predictor module
├── testing/
│   └── test_predictor.py               # Automated unit tests
├── generate_data.py                    # Synthetic dataset generator
├── train_model.py                      # Training & evaluation script
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation

from sentinelflow_ml import SentinelFlowPredictor

# Initialize the predictor (loads the model once into memory)
predictor = SentinelFlowPredictor()

# Example sensor payload from your FastAPI request body
sensor_data = {
    "pressure": 3.5,
    "flow_rate": 120.0,
    "temperature": 25.0,
    "pump_vibration": 1.2,
    "pump_current": 10.5,
    "ph": 7.2,
    "tank_level": 55.0
}

# Run prediction
result = predictor.predict(sensor_data)
print(result)
# Output: {'fault_type': 'NORMAL', 'confidence': 1.0, 'probabilities': {...}, 'status': 'success'}

python testing/test_predictor.py