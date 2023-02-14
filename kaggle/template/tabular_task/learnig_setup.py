import argparse
import mlflow

# Get parameters from argparse
def get_params_from_commandline():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, required=True, 
                        help="Path to input dataset")
    parser.add_argument("preprocess_path", type=str, required=True, 
                        help="Path to co")
    parser.add_argument("-rs", "--random_state", type=int, default=42, required=False, 
                        help="Random State")
    parser.add_argument("-nf", "--num_folds", type=int, default=5, required=False, 
                        help="Number of folds in CV (default: 5, leave-one-out: -1)")
    params = parser.parse_args()

# manage paramaters
class LearningSetup:
    
    def __init__(self):
        mlflow
        
    def save_results(self):
        
    def 
    
    
    
# Generate ipynb file for notebook competition
class GenerateNotebook:
    