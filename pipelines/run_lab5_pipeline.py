import subprocess, sys
for module in ['src.validate_data', 'src.preprocess_pipeline', 'src.validate_outputs']:
    res = subprocess.run([sys.executable, '-m', module])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] pipelines/run_lab5_pipeline.py Finished!")
