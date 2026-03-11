import os
import subprocess
import sys

def run_step(script_name):
    print(f"\n{'='*20} Running {script_name} {'='*20}")
    result = subprocess.run([sys.executable, script_name], capture_output=False)
    if result.returncode != 0:
        print(f"Error running {script_name}. Exiting.")
        sys.exit(1)

def main():
    print("Starting Fake Account Detection Pipeline...")
    
    run_step('data_generation.py')
    run_step('preprocessing.py')
    run_step('eda.py')
    run_step('train_model.py')
    
    print("\n" + "="*50)
    print("Pipeline Completed Successfully!")
    print("Check the 'plots' directory for visualizations.")
    print("="*50)

if __name__ == "__main__":
    main()
