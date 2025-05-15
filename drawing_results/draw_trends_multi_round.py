import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy

def calculate_phi_multi_round(prediction_file_path, model_list):
    with open(prediction_file_path, 'r') as f:
        data = json.load(f)

    round_number = len(data[0]["predictions"])
    joint_posterior_list = [[] for i in range(round_number)]
    for news_dict in data:
        predictions = news_dict["predictions"]
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
    for round_idx, round_joint_posterior_list in enumerate(joint_posterior_list):
        number_of_signals = len(round_joint_posterior_list)
        for joint_posterior in round_joint_posterior_list:
            phi_list[round_idx][joint_posterior.tobytes()] = phi_list[round_idx].get(joint_posterior.tobytes(), 0) + 1/number_of_signals
    
    return phi_list


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

def create_trend(consistency_metric_list, phi_list, output_file_path):
    round_number = len(consistency_metric_list)
    
    js_divergence_list = []
    for phi in phi_list:
        js_divergence = 0
        for k in phi.keys():
            joint_posterior = np.frombuffer(k, dtype=np.float32).reshape(2, 2)
            # JS divergence
            js = scipy.spatial.distance.jensenshannon(joint_posterior[0], joint_posterior[1])
            js_divergence += phi[k] * js
        js_divergence_list.append(js_divergence)

    # Plot configuration
    plt.figure(figsize=(12, 10))

    plt.plot(consistency_metric_list, label="Consistency Metric")
    plt.plot(js_divergence_list, label="JS Divergence")

    plt.xticks(range(round_number))
    plt.xlabel('Round')
    plt.ylabel('Consistency Metric')
    plt.legend()

    plt.title(f'Consistency Metrics', pad=20)
    
    # Save figure
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    experiment_name = "2025-04-28_17-17-45"
    result_file_path = f'./results/consistency_metric/Multi_Round/{experiment_name}.txt'
    LLM_prediction_file_path = f'./results/LLM_predictions/Multi_Round/{experiment_name}.json'

    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']
    
    consistency_metric_data = parse_consistency_files(result_file_path)

    for agent1 in model_list:
        for agent2 in model_list:
            agents = [agent1, agent2]
            number_of_agents_pair = len(agents)
            phi_pair_list = calculate_phi_multi_round(LLM_prediction_file_path, agents)

            data = consistency_metric_data[(agent1, agent2)]

            agent1_for_path = agent1
            if '/' in agent1_for_path:
                agent1_for_path = "-".join(agent1_for_path.split('/'))
            agent2_for_path = agent2
            if '/' in agent2_for_path:
                agent2_for_path = "-".join(agent2_for_path.split('/'))
            
            os.makedirs(f'./results/figures/Multi_Round/trend/{experiment_name}', exist_ok=True)
            output_file_path = f'./results/figures/Multi_Round/trend/{experiment_name}/{agent1_for_path}_{agent2_for_path}.png'

            create_trend(data, phi_pair_list, output_file_path)
