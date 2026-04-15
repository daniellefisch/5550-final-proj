from codecarbon import EmissionsTracker
import subprocess
import time

scripts = [
    "src/model_linear.py",
    "src/model_random_forest.py",
    "src/model_gradient_boosting.py",
    "src/feature_importance.py",
]

if __name__ == "__main__":
    tracker = EmissionsTracker(
        measure_power_secs=1,
        log_level="error"
    )
    start_time = time.time()

    tracker.start()

    for script in scripts:
        print(f"\nRunning {script}...")
        result = subprocess.run(["python", script], check=True)
    
    emissions = tracker.stop()
    end_time = time.time()

    runtime_minutes = (end_time - start_time) / 60

    print(f"\nTotal runtime: {runtime_minutes:.2f} minutes")
    print(f"Estimated CO2 emissions: {emissions:.6f} kg")