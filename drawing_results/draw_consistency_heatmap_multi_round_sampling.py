import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tikzplotlib

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def parse_consistency_files(result_file_path):
    data = {}

    with open(result_file_path, 'r') as f:
        lines = f.readlines()

        overall_flag = True
        round_number = 5
        date_data_tmp = None
        agent_0 = None
        agent_1 = None
        for line in lines:
            line = line.strip()
            if 'Agent' in line:
                agents = line.split('\t')
                if len(agents) == 2:
                    overall_flag = False
                    agent_0 = agents[0].split(" ")[-1]
                    agent_1 = agents[1].split(" ")[-1]
                    data[(agent_0, agent_1)] = [[] for i in range(round_number)]
                else:
                    agent_list = [x.split(" ")[-1] for x in line.strip().split('\t')]
                    data["agent list"] = agent_list
                    data["overall metric"] = [[] for i in range(round_number)]
            elif 'Date' in line:
                date_data_tmp = []
            elif 'Consistency Metric' in line:
                try:
                    consistency_metric = float(line.split(' ')[-1])
                    date_data_tmp.append(consistency_metric)
                except:
                    date_data_tmp.append(None)
            else:
                if date_data_tmp is not None and None not in date_data_tmp:
                    
                    # print(f"Agent 0: {agent_0}, Agent 1: {agent_1}")
                    # print(f"Date Data: {date_data_tmp}")
                    # print("\n")

                    for i in range(round_number):
                        if overall_flag:
                            data["overall metric"][i].append(date_data_tmp[i])
                        else:
                            data[(agent_0, agent_1)][i].append(date_data_tmp[i])
                    date_data_tmp = None

    return data

def create_heatmap(data, output_dir, experiment_name):
    """Generate and save a heatmap from parsed data""" 
    agent_list = data['agent list']
    overall_metric = data["overall metric"]

    print("------")
    print(f"Agent List: {agent_list}")
    print(f"Overall Metric: {overall_metric}")
    print("------")
    
    n = len(agent_list)
    round_number = len(overall_metric)
    matrix = [np.full((n, n), np.nan) for i in range(round_number)]

    for round_idx in range(round_number):
        for i in range(n):
            for j in range(n):
                matrix[round_idx][(i, j)] = sum(data[(agent_list[i], agent_list[j])][round_idx]) / len(data[(agent_list[i], agent_list[j])][round_idx])

        # Plot configuration
        plt.figure(figsize=(12, 10))
        sns.set(font_scale=0.8)
    
        # Create diverging colormap
        max_val = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
        # Plot heatmap
        ax = sns.heatmap(matrix[round_idx], annot=True, fmt=".4f", cmap=cmap,
                     center=0, vmin=-max_val, vmax=max_val,
                     mask=np.isnan(matrix[round_idx]), square=True,
                     linewidths=0.5, linecolor='#444', cbar_kws={"shrink": 0.8})
    
        # Set labels and title
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels(agent_list, rotation=45, ha='right')
        ax.set_yticklabels(agent_list, rotation=0)
        plt.title(f'Consistency Metrics', pad=20)
    
        # Save figure
        output_file_path = os.path.join(output_dir, f"{experiment_name}_Round{round_idx}.png")
        plt.savefig(output_file_path, bbox_inches='tight')

        tikz_path = f'./results/tikz/Multi_Round_Sampling/confusion_matrix/{experiment_name}/{experiment_name}_Round{round_idx}.tikz'
        tikzplotlib.save(tikz_path)
        plt.close()

if __name__ == "__main__":
    experiment_name = "2025-05-05_17-14-05"
    result_file_path = f'./results/consistency_metric/Multi_Round_Sampling/{experiment_name}.txt'
    output_dir = f'./results/figures/Multi_Round_Sampling/confusion_matrix/{experiment_name}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'./results/tikz/Multi_Round_Sampling/confusion_matrix/{experiment_name}' , exist_ok=True)

    data = parse_consistency_files(result_file_path)
    
    create_heatmap(data, output_dir, experiment_name)
