import os
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

result_dict = {}
result_file_path = './results/consistency_metric/json_format_response.txt'

with open(result_file_path, 'r') as f:
    agent_0 = None
    agent_1 = None
    consistency_metric = 0
    lines = f.readlines()
    lines = lines[2:]
    for line in lines:
        line = line.strip()
        print(line)
        if 'Consistency Metric' in line:
            consistency_metric = float(line.split(' ')[-1])
            result_dict[(agent_0, agent_1)] = consistency_metric
        else:
            agents = line.split('\t')
            agent_0 = agents[0].split(" ")[-1]
            agent_1 = agents[1].split(" ")[-1]

model_list = ['gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4-turbo-2024-04-09', 'gpt-4-0613', 'gpt-3.5-turbo-0125']
mapping_names = {'gpt-4o-2024-05-13': "4o",
                 'gpt-4o-mini-2024-07-18': "4o-mini",
                 'gpt-4-turbo-2024-04-09': "4-turbo",
                 'gpt-4-0613': "4",
                 'gpt-3.5-turbo-0125': "3.5"}


consistency = []
for y in model_list:
    tmp = []
    for x in model_list:
        if (x, y) in result_dict.keys():
            tmp.append(result_dict[(x, y)])
        elif (y, x) in result_dict.keys():
            tmp.append(result_dict[(y, x)])
        else:
            print(f"Error! x: {x}, y: {y}")
    consistency.append(tmp)

fig, ax = plt.subplots()
im = ax.imshow(consistency, cmap='summer')

# Show axis ticks and labels  
ax.set_xticks(np.arange(len(model_list)))
ax.set_yticks(np.arange(len(model_list)))

x_labels = [mapping_names[x] for x in model_list]
y_labels = [mapping_names[y] for y in model_list]

ax.set_xticklabels(x_labels)
ax.set_yticklabels(y_labels)

# Set color bar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.set_ylabel('Consistency Metric', rotation=-90, va="bottom")

# tikz_path = os.path.join('./result_images', 'consistency_metric_all.tikz')
# tikzplotlib.save(tikz_path)
plt.savefig("results/images/confusion_matrix.png")
plt.show()
