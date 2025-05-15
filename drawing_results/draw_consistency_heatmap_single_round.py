import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def parse_consistency_files(result_file_path):
    data = {}

    with open(result_file_path, 'r') as f:
        lines = f.readlines()

        agent_list = [x.split(" ")[-1] for x in lines[0].strip().split('\t')]
        data["agent list"] = agent_list
        overall_metric = float(lines[1].strip().split(' ')[-1])
        data["overall metric"] = overall_metric

        agent_0 = None
        agent_1 = None
        consistency_metric = 0

        lines = lines[2:]
        for line in lines:
            line = line.strip()
            # print(line)
            if 'Consistency Metric' in line:
                consistency_metric = float(line.split(' ')[-1])
                data[(agent_0, agent_1)] = consistency_metric
            else:
                agents = line.split('\t')
                agent_0 = agents[0].split(" ")[-1]
                agent_1 = agents[1].split(" ")[-1]

    return data

def create_heatmap(data, output_file_path):
    """Generate and save a heatmap from parsed data""" 
    agent_list = data['agent list']
    overall_metric = data["overall metric"]

    print("------")
    print(f"Agent List: {agent_list}")
    print(f"Overall Metric: {overall_metric}")
    print("------")
    
    n = len(agent_list)
    matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(n):
            matrix[(i, j)] = data[(agent_list[i], agent_list[j])]

    # Plot configuration
    plt.figure(figsize=(12, 10))
    sns.set(font_scale=0.8)
    
    # Create diverging colormap
    max_val = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
    # Plot heatmap
    ax = sns.heatmap(matrix, annot=True, fmt=".4f", cmap=cmap,
                     center=0, vmin=-max_val, vmax=max_val,
                     mask=np.isnan(matrix), square=True,
                     linewidths=0.5, linecolor='#444', cbar_kws={"shrink": 0.8})
    
    # Set labels and title
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(agent_list, rotation=45, ha='right')
    ax.set_yticklabels(agent_list, rotation=0)
    plt.title(f'Consistency Metrics', pad=20)
    
    # Save figure
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    experiment_name = "2025-04-20_23-07-21"
    result_file_path = f'./results/consistency_metric/Single_Round/{experiment_name}.txt'
    output_file_path = f'./results/figures/Single_Round/{experiment_name}.png'

    data = parse_consistency_files(result_file_path)
    create_heatmap(data, output_file_path)
