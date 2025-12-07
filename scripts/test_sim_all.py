#!/usr/bin/env python
import subprocess, sys

def sh(cmd):
    print("+", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    sh("python scripts/test_sim_med2vec_baseline.py")
    sh("python scripts/test_sim_plus_model.py")
    sh("python scripts/test_sim_ml_baselines.py")

if __name__ == "__main__":
    main()
