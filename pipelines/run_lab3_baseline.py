import subprocess, sys
for module in ['src.preprocess', 'src.train', 'src.evaluate']:
    res = subprocess.run([sys.executable, '-m', module])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] pipelines/run_lab3_baseline.py Finished!")
