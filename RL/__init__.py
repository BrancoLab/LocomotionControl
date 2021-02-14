import sys

sys.path.append("./")
sys.setrecursionlimit(10000)

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
