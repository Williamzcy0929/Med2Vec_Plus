#!/usr/bin/env python
import subprocess, sys

def sh(cmd):
    print("+", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    sh("python -m src.med2vec_plus.train --config configs/med2vec_only.yaml")
    sh("python -m src.med2vec_plus.evaluate --config configs/med2vec_only.yaml --ckpt ckpts_med2vec_only/best.pt --split test")

if __name__ == "__main__":
    main()
