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

if __name__ == "__main__":
    # experiment_name = "2025-04-21_18-44-41"
    experiment_name = "2025-04-28_17-17-45"
    result_file_path = f'./results/consistency_metric/Multi_Round/{experiment_name}.txt'

    data = parse_consistency_files(result_file_path)
    # create_heatmap(data, output_dir, experiment_name)

    agent_list = data['agent list']
    overall_metric = data["overall metric"]

    for agent in agent_list:
        avg_consistency_metric = 0
        for agent2 in agent_list:
            if agent2 == agent:
                continue
            avg_consistency_metric += data[(agent, agent2)][0]
        avg_consistency_metric /= (len(agent_list) - 1)


        print(f"Agent: {agent}")
        print(f"Average Consistency Metric: {avg_consistency_metric}")
        print(f"Within Consistency Metric: {within_model_consistency[agent]}")

