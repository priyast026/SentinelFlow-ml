import sys
import os

# Add root directory to python path so we can import sentinelflow_ml
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentinelflow_ml import SentinelFlowPredictor

def test_prediction_module():
    print("Initializing SentinelFlowPredictor...")
    predictor = SentinelFlowPredictor()
    
    # Test case 1: Normal operating sensor values
    normal_sample = {
        "pressure": 3.5,
        "flow_rate": 120.0,
        "temperature": 25.0,
        "pump_vibration": 1.2,
        "pump_current": 10.5,
        "ph": 7.2,
        "tank_level": 55.0
    }
    
    print("\nTesting Normal Sensor Input:")
    result_normal = predictor.predict(normal_sample)
    print(result_normal)
    assert result_normal["fault_type"] == "NORMAL", f"Expected NORMAL, got {result_normal['fault_type']}"

    # Test case 2: Pipe Blockage sensor values (High pressure, low flow)
    blockage_sample = {
        "pressure": 6.2,
        "flow_rate": 45.0,
        "temperature": 28.0,
        "pump_vibration": 2.1,
        "pump_current": 14.2,
        "ph": 7.2,
        "tank_level": 30.0
    }
    
    print("\nTesting Pipe Blockage Sensor Input:")
    result_blockage = predictor.predict(blockage_sample)
    print(result_blockage)
    assert result_blockage["fault_type"] == "PIPE_BLOCKAGE", f"Expected PIPE_BLOCKAGE, got {result_blockage['fault_type']}"

    print("\nAll predictor unit tests passed successfully!")

if __name__ == "__main__":
    test_prediction_module()