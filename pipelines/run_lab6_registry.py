import subprocess, sys
for module in ['src.train_registry', 'src.automate_lifecycle', 'src.generate_registry_report']:
    res = subprocess.run([sys.executable, '-m', module])
    if res.returncode != 0:
        sys.exit(res.returncode)
print("[SUCCESS] pipelines/run_lab6_registry.py Finished!")
