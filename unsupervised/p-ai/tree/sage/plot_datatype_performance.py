import matplotlib.pyplot as plt
import pandas as pd, numpy as np

data = pd.read_csv("perf_v2.csv")

unique_ops = np.unique(data["operation"])

for op in unique_ops:
    df = data[data["operation"]==op]
    plt.plot(df["num_reads"],df["t_per_read_s"],label=op)
plt.xlabel("LC reads requested")
plt.ylabel("Time per read")
plt.legend()
plt.show()