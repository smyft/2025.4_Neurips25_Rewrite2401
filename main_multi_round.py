import json
import numpy as np
import os

from calculating_metrics.consistency_metric import calculate_consistency_metric

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

if __name__ == "__main__":
    L = 2  # number of world states
    # model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'deepseek/deepseek-r1', 'anthropic/claude-3.7-sonnet']
    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']

    experiment_name = "2025-04-28_17-17-45"
    log_file_path = f'./results/consistency_metric/Multi_Round/{experiment_name}.txt'
    LLM_prediction_file_path = f'./results/LLM_predictions/Multi_Round/{experiment_name}.json'

    with open(log_file_path, 'w') as f:
        # Overall consistency metric
        phi_list = calculate_phi_multi_round(LLM_prediction_file_path, model_list)
        number_of_agents = len(model_list)
        consistency_metric_list = []
        for phi in phi_list:
            consistency_metric = calculate_consistency_metric(phi, number_of_agents, 2)
            consistency_metric_list.append(consistency_metric)
        
        print(f"Models: {model_list}")
        print(f"Consistency Metric List: {consistency_metric_list}")

        for idx, model_name in enumerate(model_list):
            f.write(f"Agent {idx}: {model_name}\t")
        f.write('\n')
        for round_idx, consistency_metric in enumerate(consistency_metric_list):
            f.write(f'Round: {round_idx}, Consistency Metric: {consistency_metric}\n')

        # Pairwise consistency metrics
        for agent1 in model_list:
            for agent2 in model_list:
                agents = [agent1, agent2]
                number_of_agents_pair = len(agents)
                phi_pair_list = calculate_phi_multi_round(LLM_prediction_file_path, agents)
                consistency_metric_pair_list = []
                for phi_pair in phi_pair_list:
                    consistency_metric_pair = calculate_consistency_metric(phi_pair, number_of_agents_pair, 2)
                    consistency_metric_pair_list.append(consistency_metric_pair)

                print(f"Models: {agents}")
                print(f"Consistency Metric List: {consistency_metric_pair_list}")

                for idx, model_name in enumerate(agents):
                    f.write(f"Agent {idx}: {model_name}\t")
                f.write('\n')
                for round_idx, consistency_metric_pair in enumerate(consistency_metric_pair_list):
                    f.write(f'Round: {round_idx}, Consistency Metric: {consistency_metric_pair}\n')
