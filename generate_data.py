import os
import numpy as np
import pandas as pd

# Set a fixed random seed so your dataset is 100% reproducible every time you run it
np.random.seed(42)

def generate_scada_dataset(num_records=10000):
    print(f"Generating synthetic SCADA dataset with {num_records} records...")
    
    # We will distribute the 10,000 records across our 7 target classes:
    # 1. NORMAL (60% - realistic industrial operating condition)
    # 2-7. Fault classes (approx 6.67% each)
    classes = [
        "NORMAL", 
        "PIPE_LEAKAGE", 
        "PIPE_BLOCKAGE", 
        "PUMP_FAILURE", 
        "PUMP_OVERLOAD", 
        "TANK_OVERFLOW", 
        "ABNORMAL_PH"
    ]
    
    # Allocate record counts per class
    class_counts = {
        "NORMAL": 6000,
        "PIPE_LEAKAGE": 666,
        "PIPE_BLOCKAGE": 666,
        "PUMP_FAILURE": 666,
        "PUMP_OVERLOAD": 666,
        "TANK_OVERFLOW": 666,
        "ABNORMAL_PH": 666
    }
    
    # Adjust last class to hit exactly 10,000 total records due to rounding
    total_allocated = sum(class_counts.values())
    if total_allocated < num_records:
        class_counts["ABNORMAL_PH"] += (num_records - total_allocated)

    data_frames = []

    for fault_type, count in class_counts.items():
        if fault_type == "NORMAL":
            # Normal operating conditions (stable ranges)
            pressure = np.random.normal(loc=3.5, scale=0.3, size=count)         # bar
            flow_rate = np.random.normal(loc=120.0, scale=5.0, size=count)    # L/min
            temperature = np.random.normal(loc=25.0, scale=1.5, size=count)   # °C
            pump_vibration = np.random.normal(loc=1.2, scale=0.2, size=count) # mm/s
            pump_current = np.random.normal(loc=10.5, scale=0.8, size=count)  # A
            ph = np.random.normal(loc=7.2, scale=0.2, size=count)             # pH units
            tank_level = np.random.normal(loc=55.0, scale=5.0, size=count)    # %

        elif fault_type == "PIPE_LEAKAGE":
            # Leakage: Drop in pressure, drop in effective flow, increased pump work
            pressure = np.random.normal(loc=1.8, scale=0.3, size=count)
            flow_rate = np.random.normal(loc=85.0, scale=6.0, size=count)
            temperature = np.random.normal(loc=25.5, scale=1.5, size=count)
            pump_vibration = np.random.normal(loc=1.5, scale=0.3, size=count)
            pump_current = np.random.normal(loc=11.0, scale=0.9, size=count)
            ph = np.random.normal(loc=7.2, scale=0.2, size=count)
            tank_level = np.random.normal(loc=40.0, scale=6.0, size=count)

        elif fault_type == "PIPE_BLOCKAGE":
            # Blockage: High restriction causes high pressure buildup and low flow rate
            pressure = np.random.normal(loc=6.2, scale=0.4, size=count)
            flow_rate = np.random.normal(loc=45.0, scale=4.0, size=count)
            temperature = np.random.normal(loc=28.0, scale=2.0, size=count)
            pump_vibration = np.random.normal(loc=2.1, scale=0.4, size=count)
            pump_current = np.random.normal(loc=14.2, scale=1.0, size=count)
            ph = np.random.normal(loc=7.2, scale=0.2, size=count)
            tank_level = np.random.normal(loc=30.0, scale=5.0, size=count)

        elif fault_type == "PUMP_FAILURE":
            # Pump Failure: Pressure drops to zero/near zero, flow stops, current drops
            pressure = np.random.normal(loc=0.4, scale=0.1, size=count)
            flow_rate = np.random.normal(loc=5.0, scale=2.0, size=count)
            temperature = np.random.normal(loc=24.0, scale=1.0, size=count)
            pump_vibration = np.random.normal(loc=0.2, scale=0.1, size=count)
            pump_current = np.random.normal(loc=0.8, scale=0.2, size=count)
            ph = np.random.normal(loc=7.2, scale=0.2, size=count)
            tank_level = np.random.normal(loc=20.0, scale=4.0, size=count)

        elif fault_type == "PUMP_OVERLOAD":
            # Pump Overload: High current spike, high vibration, high temperature
            pressure = np.random.normal(loc=4.8, scale=0.5, size=count)
            flow_rate = np.random.normal(loc=130.0, scale=8.0, size=count)
            temperature = np.random.normal(loc=38.5, scale=2.5, size=count)
            pump_vibration = np.random.normal(loc=4.5, scale=0.6, size=count)
            pump_current = np.random.normal(loc=22.5, scale=1.5, size=count)
            ph = np.random.normal(loc=7.2, scale=0.2, size=count)
            tank_level = np.random.normal(loc=60.0, scale=6.0, size=count)

        elif fault_type == "TANK_OVERFLOW":
            # Tank Overflow: Tank level dangerously high (>90%), normal flow parameters otherwise
            pressure = np.random.normal(loc=3.5, scale=0.3, size=count)
            flow_rate = np.random.normal(loc=115.0, scale=6.0, size=count)
            temperature = np.random.normal(loc=25.0, scale=1.5, size=count)
            pump_vibration = np.random.normal(loc=1.2, scale=0.2, size=count)
            pump_current = np.random.normal(loc=10.5, scale=0.8, size=count)
            ph = np.random.normal(loc=7.2, scale=0.2, size=count)
            tank_level = np.random.normal(loc=95.0, scale=2.5, size=count)

        elif fault_type == "ABNORMAL_PH":
            # Abnormal pH: pH level shifted out of neutral range (either overly acidic or alkaline)
            pressure = np.random.normal(loc=3.5, scale=0.3, size=count)
            flow_rate = np.random.normal(loc=120.0, scale=5.0, size=count)
            temperature = np.random.normal(loc=26.0, scale=1.5, size=count)
            pump_vibration = np.random.normal(loc=1.2, scale=0.2, size=count)
            pump_current = np.random.normal(loc=10.5, scale=0.8, size=count)
            # Randomly split between highly acidic (< 5.5) or highly alkaline (> 9.0)
            ph_values = np.where(
                np.random.rand(count) > 0.5,
                np.random.normal(loc=4.5, scale=0.4, size=count),
                np.random.normal(loc=10.2, scale=0.5, size=count)
            )
            ph = ph_values
            tank_level = np.random.normal(loc=55.0, scale=5.0, size=count)

        # Create a temporary DataFrame for this specific fault class
        df_class = pd.DataFrame({
            'pressure': np.round(pressure, 3),
            'flow_rate': np.round(flow_rate, 3),
            'temperature': np.round(temperature, 3),
            'pump_vibration': np.round(pump_vibration, 3),
            'pump_current': np.round(pump_current, 3),
            'ph': np.round(ph, 3),
            'tank_level': np.round(tank_level, 3),
            'fault_type': fault_type
        })
        
        data_frames.append(df_class)

    # Combine all classes together
    master_df = pd.concat(data_frames, ignore_index=True)
    
    # Shuffle the dataset so rows are in random order (crucial for machine learning)
    master_df = master_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Ensure the target directory 'dataset' exists
    os.makedirs("dataset", exist_ok=True)
    
    # Save to CSV
    output_path = os.path.join("dataset", "scada_synthetic_dataset.csv")
    master_df.to_csv(output_path, index=False)
    
    print(f"Dataset successfully generated and saved to: {output_path}")
    print(f"Total rows generated: {len(master_df)}")
    print("\nClass distribution in generated dataset:")
    print(master_df['fault_type'].value_counts())

if __name__ == "__main__":
    generate_scada_dataset(10000)