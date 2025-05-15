import json
import numpy as np
import os

from calculating_metrics.consistency_metric import calculate_consistency_metric

def calculate_phi_multi_round(prediction_file_path):
    with open(prediction_file_path, 'r') as f:
        data = json.load(f)

    round_number = len(data[0]["predictions"])
    joint_posterior_list = [[] for i in range(round_number)]
    phi_dict = [dict() for i in range(round_number)]

    for news_dict in data:
        for round_dict in news_dict["predictions"]:
            error_data = False
            round_idx = round_dict["round_idx"]
            model_predictions = round_dict["model_predictions"]

            joint_posterior = []
            for prediction_dict in model_predictions:
                rise_prob = prediction_dict["rise_prediction"]
                fall_prob = prediction_dict["fall_prediction"]
                if (rise_prob + fall_prob) < 0.95:
                    error_data = True

                posterior = np.array([rise_prob, 1-rise_prob], dtype=np.float32)
                joint_posterior.append(posterior)

            joint_posterior = np.stack(joint_posterior, axis=0)
            if error_data == False:
                joint_posterior_list[round_idx].append(joint_posterior)

    for i in range(round_number):
        number_of_signals = len(joint_posterior_list[i])
        for joint_posterior in joint_posterior_list[i]:
            phi_dict[i][joint_posterior.tobytes()] = phi_dict[i].get(joint_posterior.tobytes(), 0) + 1/number_of_signals
    
    return phi_dict

if __name__ == "__main__":
    L = 2 # number of world states
    model_list = ['gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4-turbo-2024-04-09', 'gpt-4-0613', 'gpt-3.5-turbo-0125']
    
    log_file_path = './results/consistency_metric/multi_rounds_json_format_response.txt'
    LLM_prediction_file_path = './results/LLM_predictions/multi_rounds_json_format_response.json'

    with open(log_file_path, 'w') as f:
        phi_dict = calculate_phi_multi_round(LLM_prediction_file_path)
        
        print("------")
        print("Joint posterior beliefs distribution: phi")
        for round_idx, phi in enumerate(phi_dict):
            print(f"Round {round_idx}:")
            for k in phi.keys():
                print(f"f: {np.frombuffer(k, dtype=np.float32).reshape(len(model_list), L)}")
                print(f"phi(f): {phi[k]}")
        print("------")
        
        number_of_agents = len(model_list)

        metrics = []

        for round_idx, phi in enumerate(phi_dict):
            consistency_metric = calculate_consistency_metric(phi, number_of_agents, 2)
            metrics.append(consistency_metric)

            print(f"Round {round_idx}:")
            print(f"Consistency Metric: {consistency_metric}")

        for idx, model_name in enumerate(model_list):
            f.write(f"Agent {idx}: ")
            f.write(model_name)
            f.write('\t')
        f.write('\n')
        f.write(f'Consistency Metric\n')
        for round_idx, metric in enumerate(metrics):
            f.write(f"Round {round_idx}: {metric}\n")
