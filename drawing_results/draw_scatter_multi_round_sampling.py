import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy

import tikzplotlib

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

def calculate_phi_multi_round_sampling(prediction_file_path, model_list):
    with open(prediction_file_path, 'r') as f:
        data = json.load(f)

    results = []
    for company_dict in data:
        company_predictions = company_dict["predictions"]
        for date_dict in company_predictions:
            date = date_dict["date"]
            sample_predictions = date_dict["predictions"]
            
            round_number = len(sample_predictions[0]["predictions"])
            joint_posterior_list = [[] for i in range(round_number)]
            for sample_dict in sample_predictions:
                sample_idx = sample_dict["Sample"]
                predictions = sample_dict["predictions"]

                for round_idx, round_predictions in enumerate(predictions):  # Process each round separately
                    error_data = False
                    joint_posterior = []
                    for model in model_list:
                        for prediction_dict in round_predictions:
                            if prediction_dict["model_name"] != model:
                                continue
                            rise_prob = prediction_dict["rise_prediction"]
                            fall_prob = prediction_dict["fall_prediction"]
                            if isinstance(rise_prob, float) and isinstance(fall_prob, float) and abs(rise_prob + fall_prob - 1.0) < 1e-8:
                                posterior = np.array([rise_prob, fall_prob], dtype=np.float32)
                                joint_posterior.append(posterior)
                            else:
                                error_data = True

                    if not error_data:
                        joint_posterior = np.stack(joint_posterior, axis=0)
                        joint_posterior_list[round_idx].append(joint_posterior)

            phi_list = [{} for i in range(round_number)]
            for round_idx, round_joint_posterior in enumerate(joint_posterior_list):
                number_of_signals = len(round_joint_posterior)
                for joint_posterior in round_joint_posterior:
                    phi_list[round_idx][joint_posterior.tobytes()] = phi_list[round_idx].get(joint_posterior.tobytes(), 0) + 1/number_of_signals

            results.append({"date": date, "phi_list": phi_list})

    return results


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
                date = line.split('\t')[-1]
                date_data_tmp = []
            elif 'Consistency Metric' in line:
                try:
                    consistency_metric = float(line.split(' ')[-1])
                    date_data_tmp.append(consistency_metric)
                except:
                    date_data_tmp.append(None)
            else:
                if date_data_tmp is not None and None not in date_data_tmp:
                    for i in range(round_number):
                        if overall_flag:
                            data["overall metric"][i].append(date_data_tmp[i])
                        else:
                            data[(agent_0, agent_1)][i].append(date_data_tmp[i])
                    date_data_tmp = None

    if date_data_tmp is not None and None not in date_data_tmp:
        for i in range(round_number):
            if overall_flag:
                data["overall metric"][i].append(date_data_tmp[i])
            else:
                data[(agent_0, agent_1)][i].append(date_data_tmp[i])
        date_data_tmp = None

    return data

if __name__ == "__main__":
    experiment_name = "2025-05-05_17-14-05"
    result_file_path = f'./results/consistency_metric/Multi_Round_Sampling/{experiment_name}.txt'
    LLM_prediction_file_path = f'./results/LLM_predictions/Multi_Round_Sampling/{experiment_name}.json'

    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']
    
    consistency_metric_data = parse_consistency_files(result_file_path)

    consistency_metric_list = []
    js_divergence_list = []
    tv_distance_list = []

    for agent1 in model_list:
        for agent2 in model_list:
            # print("------")
            # print(f"Agent1: {agent1}, Agent2: {agent2}")
            agents = [agent1, agent2]
            number_of_agents_pair = len(agents)
            results = calculate_phi_multi_round_sampling(LLM_prediction_file_path, agents)

            # consistency_metric_data[(agent1, agent2)]: round_numberxdate_number
            consistency_metric_list.extend(consistency_metric_data[(agent1, agent2)][0])

            # print(f"Length of consistency metric: {len(consistency_metric_data[(agent1, agent2)][0])}")  
            
            for date_dict in results:
                date = date_dict["date"]
                phi_list = date_dict["phi_list"]
            
                error_data = False
                for phi in phi_list:
                    if phi=={}:
                        error_data = True
                if error_data:
                    # print(f"Dropping Date: {date}")
                    continue

                phi = phi_list[-1]
                js_divergence = 0
                tv_distance = 0
                for k in phi.keys():
                    joint_posterior = np.frombuffer(k, dtype=np.float32).reshape(2, 2)
                    # JS divergence
                    js = scipy.spatial.distance.jensenshannon(joint_posterior[0], joint_posterior[1])
                    tv = 0.5 * np.sum(np.abs(joint_posterior[0] - joint_posterior[1]))

                    js_divergence += phi[k] * js
                    tv_distance += phi[k] * tv

                js_divergence_list.append(js_divergence)
                tv_distance_list.append(tv_distance)

            # print("------")

    consistency_metric_list = np.array(consistency_metric_list)
    js_divergence_list = np.array(js_divergence_list)
    tv_distance_list = np.array(tv_distance_list)

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(consistency_metric_list, js_divergence_list, 1)
    x_line = np.linspace(min(consistency_metric_list), max(consistency_metric_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(consistency_metric_list, js_divergence_list, label='Data Point', color='blue')
    plt.xlabel("Consistency Metric")
    plt.ylabel("JS Divergence")
    plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round_Sampling/{experiment_name}_JSDivergence.png'
    plt.savefig(output_file_path, bbox_inches='tight')

    tikz_path = f'./results/tikz/Multi_Round_Sampling/{experiment_name}_JSDivergence.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(consistency_metric_list, tv_distance_list, 1)
    x_line = np.linspace(min(consistency_metric_list), max(consistency_metric_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(consistency_metric_list, tv_distance_list, label='Data Point', color='blue')
    plt.xlabel("Consistency Metric")
    plt.ylabel("TV Distance")
    plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round_Sampling/{experiment_name}_TVDistance.png'
    plt.savefig(output_file_path, bbox_inches='tight')

    tikz_path = f'./results/tikz/Multi_Round_Sampling/{experiment_name}_TVDivergence.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()
