import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

result_file_path = './results/consistency_metric/multi_rounds_json_format_response.txt'

metrics = []

with open(result_file_path, 'r') as f:
    lines = f.readlines()
    lines = lines[2:]
    for line in lines:
        line = line.strip()
        print(line)
        if "Round" in line:
            round_idx = int(line.split(' ')[1][0])
            consistency_metric = float(line.split(' ')[-1])
            metrics.append(consistency_metric)

plt.plot(metrics)
plt.xticks(range(len(metrics)))
plt.xlabel('Round')
plt.ylabel('Consistency Metric')
plt.savefig("results/images/trend.png")
plt.show()
