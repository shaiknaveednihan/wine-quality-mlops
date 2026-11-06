import subprocess, sys
for module in ['src.preprocess', 'src.train_mlflow', 'src.validate_reproducibility']:
    res = subprocess.run([sys.executable, '-m', module])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] pipelines/run_lab4_tracking.py Finished!")
