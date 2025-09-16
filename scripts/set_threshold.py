# # scripts/set_threshold.py
# import joblib, sys
# path, thr = sys.argv[1], float(sys.argv[2])
# bundle = joblib.load(path)
# bundle["decision_threshold"] = thr
# joblib.dump(bundle, path)
# print(f"Updated {path} with threshold={thr}")

# scripts/set_threshold.py
import sys, joblib
path, thr = sys.argv[1], float(sys.argv[2])
b = joblib.load(path)
b["decision_threshold"] = thr
joblib.dump(b, path)
print(f"Updated {path} with threshold={thr}")
