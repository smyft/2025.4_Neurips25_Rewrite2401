import json
import numpy as np
import os

from calculating_metrics.consistency_metric import calculate_consistency_metric

def calculate_phi_multi_round(temp0_file_path, temp1_file_path, model_list):
    with open(temp0_file_path, 'r') as f:
        data0 = json.load(f)
    with open(temp1_file_path, 'r') as f:
        data1 = json.load(f)

    round_number = len(data0[0]["predictions"])

    assert len(data0)==len(data1)

    joint_posterior_list = {model: [] for model in model_list}
    for news_dict0, news_dict1 in zip(data0, data1):
        predictions0 = news_dict0["predictions"][0]
        predictions1 = news_dict1["predictions"][0]

        for model in model_list:
            error_data = False
            joint_posterior = []

            for prediction_dict in predictions0:
                if prediction_dict["model_name"] != model:
                    continue
                rise_prob = prediction_dict["rise_prediction"]
                fall_prob = prediction_dict["fall_prediction"]
                if isinstance(rise_prob, float) and isinstance(fall_prob, float) and abs(rise_prob + fall_prob - 1.0) < 1e-8:
                    posterior = np.array([rise_prob, fall_prob], dtype=np.float32)
                    joint_posterior.append(posterior)
                else:
                    error_data = True
            
            for prediction_dict in predictions1:
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
                joint_posterior_list[model].append(joint_posterior)

    phi_list = {model: {} for model in model_list}
    for model in model_list:
        jp_list = joint_posterior_list[model]
        number_of_signals = len(jp_list)
        for joint_posterior in jp_list:
            phi_list[model][joint_posterior.tobytes()] = phi_list[model].get(joint_posterior.tobytes(), 0) + 1/number_of_signals
    
    return phi_list

if __name__ == "__main__":
    L = 2  # number of world states
    # model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'deepseek/deepseek-r1', 'anthropic/claude-3.7-sonnet']
    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']

    temp0_path_file = f'./results/LLM_predictions/Multi_Round/2025-04-21_18-44-41.json'
    temp1_path_file = f'./results/LLM_predictions/Multi_Round/2025-04-28_17-17-45.json'

    phi_dict = calculate_phi_multi_round(temp0_path_file, temp1_path_file, model_list)

    for model in model_list:
        phi = phi_dict[model]
        consistency_metric = calculate_consistency_metric(phi, 2, 2)

        print("------")
        print(f"Models: {model}")
        print(f"Consistency Metric List: {consistency_metric}")
        print("------\n")
