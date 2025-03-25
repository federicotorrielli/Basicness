import pandas as pd
import re
from collections import Counter
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import numpy as np
import itertools

def clean_model_name(name):
    """Convert model folder name to column name format"""
    return name.lower().replace('-', '_').replace('.', '_')

def extract_last_value(line):
    """
    Extract the last value from a line of CSV.
    Handles both simple comma splits and more complex cases.
    """
    # Try to find a score at the end of the line
    # Common patterns: ...,3 or ...,4 or ...,2
    match = re.search(r',(\d+)$', line.strip())
    if match:
        return int(match.group(1))
    
    # If no simple match, try splitting by comma
    parts = line.strip().split(',')
    if parts and len(parts) > 1:
        # Try to convert the last part to an integer
        try:
            return int(parts[-1])
        except ValueError:
            # See if there's a number in the last part
            match = re.search(r'(\d+)', parts[-1])
            if match:
                return int(match.group(1))
    
    # If all else fails, return None
    return None

def process_model_directory(model_dir):
    """
    Process all 10 CSV files in a model directory reading them line by line
    
    Args:
        model_dir (Path): Path to model directory
        
    Returns:
        dict: Dictionary mapping row index to majority vote
    """
    print(f"Processing {model_dir.name}...")
    all_scores = {}
    
    # Process all 10 CSV files
    for i in range(1, 11):
        file_path = model_dir / f"ann_{i}.csv"
        if not file_path.exists():
            print(f"Warning: File {file_path} not found")
            continue

        try:
            # Read file line by line
            with open(file_path, 'r', encoding='utf-8') as f:
                # Skip header line
                next(f, None)
                
                # Process each line
                for idx, line in enumerate(f):
                    if idx not in all_scores:
                        all_scores[idx] = []
                    
                    # Extract score from the line
                    score = extract_last_value(line)
                    if score is not None:
                        all_scores[idx].append(score)
                    else:
                        print(f"Warning: Could not extract score from line {idx+2} in {file_path}")
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Calculate majority vote for each concept
    majority_votes = {}
    for idx, scores in all_scores.items():
        if not scores:
            continue
            
        # Use Counter to find most common score
        counter = Counter(scores)
        # Get the most common value (in case of tie, take the first one)
        majority_vote = counter.most_common(1)[0][0]
        majority_votes[idx] = majority_vote
    
    return majority_votes

def calculate_model_consistency(model_dir):
    """
    Calculate self-consistency metrics for a model across its runs
    
    Args:
        model_dir (Path): Path to model directory
        
    Returns:
        dict: Dictionary with consistency metrics (kappa and percentage agreement)
    """
    print(f"Calculating consistency for {model_dir.name}...")
    
    # Store all annotations in a dict mapping file number -> {row_idx: score}
    annotations = {}
    
    # Process all CSV files
    for i in range(1, 11):
        file_path = model_dir / f"ann_{i}.csv"
        if not file_path.exists():
            continue
            
        annotations[i] = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                next(f, None)  # Skip header
                for idx, line in enumerate(f):
                    score = extract_last_value(line)
                    if score is not None:
                        annotations[i][idx] = score
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    # Calculate pairwise agreement metrics
    kappas = []
    agreements = []
    
    # Get all combinations of file pairs
    file_pairs = list(itertools.combinations(annotations.keys(), 2))
    
    for file1, file2 in file_pairs:
        # Find common indices
        common_indices = set(annotations[file1].keys()) & set(annotations[file2].keys())
        
        if not common_indices:
            continue
            
        # Create arrays for the common items
        y1 = [annotations[file1][idx] for idx in common_indices]
        y2 = [annotations[file2][idx] for idx in common_indices]
        
        # Calculate metrics
        try:
            kappa = cohen_kappa_score(y1, y2)
            agreement = sum(a == b for a, b in zip(y1, y2)) / len(y1)
            
            kappas.append(kappa)
            agreements.append(agreement)
        except Exception as e:
            print(f"Error calculating metrics for {file1} vs {file2}: {e}")
    
    # Average the metrics
    avg_kappa = np.mean(kappas) if kappas else 0
    avg_agreement = np.mean(agreements) if agreements else 0
    
    return {
        "model": model_dir.name,
        "kappa": avg_kappa,
        "agreement_pct": avg_agreement * 100
    }

def generate_consistency_report():
    """Generate a CSV report of model self-consistency metrics"""
    silver_dir = Path("silver_annotation")
    output_file = Path("model_consistency.csv")
    
    if not silver_dir.exists():
        print(f"Error: Directory {silver_dir} not found")
        return
    
    consistency_results = []
    
    for model_dir in sorted(silver_dir.iterdir()):
        if not model_dir.is_dir():
            continue
            
        consistency = calculate_model_consistency(model_dir)
        consistency_results.append(consistency)
    
    # Create DataFrame and save to CSV
    consistency_df = pd.DataFrame(consistency_results)
    consistency_df.to_csv(output_file, index=False)
    print(f"Model consistency report saved to {output_file}")

def main():
    # Define paths
    silver_dir = Path("silver_annotation")
    output_file = Path("culture_annotation_datasets_silver_annotated_simple.csv")
    complex_file = Path("culture_annotation_datasets_silver_annotated_complex.csv")
    
    # Check if the source directory exists
    if not silver_dir.exists():
        print(f"Error: Directory {silver_dir} not found")
        return
    
    # First check if the complex file exists to use as a base
    if complex_file.exists():
        print(f"Using {complex_file} as base")
        # Read the complex file to use as a template
        base_df = pd.read_csv(complex_file)
        # Use only the relevant columns for our simple file
        result_df = base_df[['ili', 'relevant_lang', 'representation_lemma', 
                             'en_definitions', 'en_most_freq_lemma', 
                             'it_most_freq_lemma', 'es_most_freq_lemma', 
                             'nb_most_freq_lemma']].copy()
    else:
        # We need to construct the base data from one of the annotation files
        # Try to find a valid annotation file to use as template
        result_df = None
        for model_dir in sorted(silver_dir.iterdir()):
            if not model_dir.is_dir():
                continue
                
            # Try each annotation file until we find one that works
            for i in range(1, 11):
                sample_file = model_dir / f"ann_{i}.csv"
                if not sample_file.exists():
                    continue
                    
                try:
                    sample_df = pd.read_csv(sample_file)
                    # Create DataFrame with all columns except the score column
                    result_df = sample_df.iloc[:, :-1].copy()
                    print(f"Using {sample_file} to create base structure")
                    break
                except:
                    continue
                    
            if result_df is not None:
                break
        
        if result_df is None:
            print("Error: Could not find a valid annotation file to use as template")
            return
            
        # Add missing columns if needed
        for col in ['ili', 'relevant_lang', 'representation_lemma']:
            if col not in result_df.columns:
                result_df[col] = ""
    
    # Check if output file exists and read it (to preserve any existing columns)
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            # Merge existing columns into our result dataframe
            for col in existing_df.columns:
                if col not in result_df.columns:
                    result_df[col] = existing_df[col]
        except Exception as e:
            print(f"Warning: Could not read existing output file: {e}")
    
    # Process each model directory
    for model_dir in sorted(silver_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        
        # Get model name and format for column name
        model_name = model_dir.name
        column_name = f"basicness_score_{clean_model_name(model_name)}"
        
        # Process model directory and get majority votes
        majority_votes = process_model_directory(model_dir)
        
        # Add or update column in result DataFrame
        for idx, value in majority_votes.items():
            if idx < len(result_df):
                result_df.loc[idx, column_name] = value
            else:
                print(f"Warning: Index {idx} out of bounds for result_df (size {len(result_df)})")
    
    # Save result to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    # Print column names to verify
    print(f"Columns in output file: {', '.join(result_df.columns)}")
    
    # Generate consistency report
    generate_consistency_report()

if __name__ == "__main__":
    main()