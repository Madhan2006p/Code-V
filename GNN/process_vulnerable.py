import os
import subprocess
from pathlib import Path
import time
import glob

# Paths
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
DATA_DIR = BASE_DIR / "Data"
VULNERABLE_DIR = DATA_DIR / "Vulnerable_funcs"
OUTPUT_DIR = DATA_DIR / "Processed"
AST_DIR = OUTPUT_DIR / "AST"
CODESENSOR_JAR = BASE_DIR.parent / "CodeSensor.jar"

def process_vulnerable_files():
    print("Scanning for vulnerable files in", VULNERABLE_DIR)
    
    # Recursively find all .txt files in Vulnerable_funcs
    # Note: They are C code but with .txt extension
    vuln_files = list(VULNERABLE_DIR.rglob("*.txt"))
    
    print(f"Found {len(vuln_files)} vulnerable files (txt extension).")
    
    if len(vuln_files) == 0:
        print("No files found. Checking logic...")
        return
        
    AST_DIR.mkdir(parents=True, exist_ok=True)
    
    processed = 0
    failed = 0
    
    for i, file_path in enumerate(vuln_files):
        # Determine output name: vulnerable_<basename>.txt
        # Result will be AST text
        output_name = f"vulnerable_{file_path.name}"
        output_path = AST_DIR / output_name
        
        # Determine if we need to process
        # (Overwriting is safer to ensure correctness)
        
        # CodeSensor expects a file path. It works on any extension if content is C?
        # Let's hope so. If not, we might need to temp rename.
        
        try:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                subprocess.run(
                    ['java', '-jar', str(CODESENSOR_JAR), str(file_path)],
                    stdout=output_file,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
            # Check if output is empty
            if output_path.stat().st_size == 0:
                print(f"Warning: Empty output for {file_path.name}")
                failed += 1
            else:
                processed += 1
                
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            failed += 1
            
        if (i+1) % 50 == 0:
            print(f"Progress: {i+1}/{len(vuln_files)}")
            
    print(f"Finished. Processed: {processed}, Failed: {failed}")

if __name__ == "__main__":
    process_vulnerable_files()
