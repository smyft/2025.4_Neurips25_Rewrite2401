import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tikzplotlib

name_map = {
    'openai/gpt-4o-mini': "GPT-4o mini",
    'google/gemini-2.0-flash-001': "Gemini 2.0 Flash",
    'google/gemini-flash-1.5': "Gemini 1.5 Flash",
    'deepseek/deepseek-chat-v3-0324': "Deepseek V3",
    'meta-llama/llama-4-scout': "Llama 4 Scout",
    'meta-llama/llama-3.3-70b-instruct': "Llama 3 70B Instruct",
    'qwen/qwen-turbo': "Qwen Turbo"
}

within_model_consistency = {
    'openai/gpt-4o-mini': -0.0024535585270432257,
    'google/gemini-2.0-flash-001': -0.005345394594999157,
    'google/gemini-flash-1.5': -0.007144902262528073,
    'deepseek/deepseek-chat-v3-0324': -0.000700280245883758,
    'meta-llama/llama-4-scout': -0.005578512824544575,
    'meta-llama/llama-3.3-70b-instruct': -0.0023381282836315645,
    'qwen/qwen-turbo': -0.0014451343145602486
}

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

        agent_list = [x.split(" ")[-1] for x in lines[0].strip().split('\t')]
        data["agent list"] = agent_list

        overall_metric = []
        round_number = 0
        while True:
            if 'Consistency Metric' in lines[round_number + 1]:
                overall_metric.append(float(lines[round_number + 1].strip().split(' ')[-1]))
                round_number += 1
            else:
                break
        data["overall metric"] = overall_metric

        agent_0 = None
        agent_1 = None
        consistency_metric = 0

        lines = lines[round_number + 1:]
        for line in lines:
            line = line.strip()
            # print(line)
            if 'Consistency Metric' in line:
                consistency_metric = float(line.split(' ')[-1])
                data[(agent_0, agent_1)].append(consistency_metric)
            else:
                agents = line.split('\t')
                agent_0 = agents[0].split(" ")[-1]
                agent_1 = agents[1].split(" ")[-1]
                data[(agent_0, agent_1)] = []

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
                matrix[round_idx][(i, j)] = data[(agent_list[i], agent_list[j])][round_idx]

                # if i == j:
                #     matrix[round_idx][(i, j)] = within_model_consistency[agent_list[i]]


        # plt.rcParams['font.family'] = 'serif' # Use the generic serif family
        # plt.rcParams['font.serif'] = ['Arial', 'Times New Roman', 'DejaVu Serif', 'DejaVuLGC Serif', 'Bitstream Vera Serif', 'Computer Modern Roman', 'Liberation Serif', 'STIXGeneral', 'Nimbus Roman', 'FreeSerif']
        plt.rcParams['font.family'] = 'Calibri'
        latex_font_size_pt = 10 # Or 11, or 12, depending on your LaTeX document
        plt.rcParams['font.size'] = latex_font_size_pt
        # These sizes are often derived relative to font.size in rcParams, but
        # can be set explicitly if needed, often useful for tick labels or annotations.
        # Matplotlib tries to interpret these relative to the LaTeX font scale.
        # plt.rcParams['axes.labelsize'] = latex_font_size_pt
        # plt.rcParams['xtick.labelsize'] = latex_font_size_pt * 0.9 # Slightly smaller ticks? Adjust as needed
        # plt.rcParams['ytick.labelsize'] = latex_font_size_pt * 0.9 # Adjust as needed
        # plt.rcParams['legend.fontsize'] = latex_font_size_pt # If you had a legend
        # plt.rcParams['figure.titlesize'] = latex_font_size_pt * 1.2 # For plt.suptitle
        # plt.rcParams['axes.titlesize'] = latex_font_size_pt # For ax.set_title
        # Annotation font size for heatmap numbers - often slightly smaller than ticks
        # Matplotlib might interpret this size differently when usetex=True; trial and error might be needed.
        annot_font_size = latex_font_size_pt * 0.9 # Adjust as needed



        # Plot configuration
        plt.figure(figsize=(9, 5))
        sns.set(font_scale=0.8)
    
        # Create diverging colormap
        max_val = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
    
        # Plot heatmap
        ax = sns.heatmap(matrix[round_idx], annot=True, fmt=".4f", cmap=cmap,
                     center=0, vmin=-max_val, vmax=0,
                     mask=np.isnan(matrix[round_idx]), square=True,
                     linewidths=0.5, linecolor='#444', cbar_kws={"shrink": 0.8},
                     annot_kws={"fontsize": annot_font_size})
    
        # Set labels and title
        ax.set_xticks(np.arange(n) + 0.5)
        ax.set_yticks(np.arange(n) + 0.5)
        ax.set_xticklabels([name_map[agent] for agent in agent_list], rotation=45, ha='right') # rotation=45, ha='right'
        ax.set_yticklabels([name_map[agent] for agent in agent_list], rotation=0)
        # plt.title(f'Consistency Metrics', pad=20)
    
        # Save figure
        output_file_path = os.path.join(output_dir, f"{experiment_name}_Round{round_idx}.pdf")
        plt.savefig(output_file_path, bbox_inches='tight')

        tikz_path = f'./results/tikz/Multi_Round/confusion_matrix/{experiment_name}/{experiment_name}_Round{round_idx}.tikz'
        tikzplotlib.save(tikz_path)
        plt.close()

if __name__ == "__main__":
    experiment_name = "2025-04-21_18-44-41"
    # experiment_name = "2025-04-28_17-17-45"
    result_file_path = f'./results/consistency_metric/Multi_Round/{experiment_name}.txt'
    output_dir = f'./results/figures/Multi_Round/confusion_matrix/{experiment_name}'

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'./results/tikz/Multi_Round/confusion_matrix/{experiment_name}' , exist_ok=True)

    data = parse_consistency_files(result_file_path)
    create_heatmap(data, output_dir, experiment_name)
