import os
import joblib
import pandas as pd
from typing import Dict, Any

class SentinelFlowPredictor:
    _instance = None
    _model_artifact = None

    def __new__(cls, model_path: str = None):
        # Singleton pattern so the model is loaded ONCE in memory, 
        # avoiding reload overhead for every incoming request in FastAPI.
        if cls._instance is None:
            cls._instance = super(SentinelFlowPredictor, cls).__new__(cls)
            cls._instance._initialize_model(model_path)
        return cls._instance

    def _initialize_model(self, model_path: str = None):
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "sentinelflow_model.joblib")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model file not found at: {model_path}. Please run train_model.py first.")
        
        print(f"Loading SentinelFlow model from {model_path}...")
        self._model_artifact = joblib.load(model_path)
        self.pipeline = self._model_artifact["pipeline"]
        self.feature_names = self._model_artifact["feature_names"]
        self.classes = self._model_artifact["classes"]
        print("Model loaded successfully into memory.")

    def predict(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts a dictionary of sensor readings, validates required features,
        and returns predicted fault category, confidence score, and probabilities.
        """
        missing_features = [f for f in self.feature_names if f not in sensor_data]
        if missing_features:
            raise ValueError(f"Missing required sensor features: {missing_features}")

        input_df = pd.DataFrame([[sensor_data[f] for f in self.feature_names]], columns=self.feature_names)

        prediction = self.pipeline.predict(input_df)[0]
        probabilities = self.pipeline.predict_proba(input_df)[0]
        
        prob_dict = {self.classes[i]: float(probabilities[i]) for i in range(len(self.classes))}
        confidence = float(max(probabilities))

        return {
            "fault_type": str(prediction),
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "status": "success"
        }