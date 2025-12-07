#!/usr/bin/env python
import subprocess, sys

def sh(cmd):
    print("+", cmd)
    r = subprocess.run(cmd, shell=True)
    if r.returncode != 0:
        sys.exit(r.returncode)

def main():
    sh("python -m src.med2vec_plus.train --config configs/default.yaml")
    sh("python -m src.med2vec_plus.evaluate --config configs/default.yaml --ckpt ckpts/best.pt --split test")

if __name__ == "__main__":
    main()
