import json
import numpy as np
import os

from calculating_metrics.consistency_metric import calculate_consistency_metric

def calculate_phi_single_round(prediction_file_path, model_list):
    with open(prediction_file_path, 'r') as f:
        data = json.load(f)

    joint_posterior_list = []
    for news_dict in data:
        error_data = False
        predictions = news_dict["predictions"]
        joint_posterior = []
        for model in model_list:
            for prediction_dict in predictions:
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
            joint_posterior_list.append(joint_posterior)

    number_of_signals = len(joint_posterior_list)

    phi = dict()
    for joint_posterior in joint_posterior_list:
        phi[joint_posterior.tobytes()] = phi.get(joint_posterior.tobytes(), 0) + 1/number_of_signals
    
    return phi
        

if __name__ == "__main__":
    L = 2 # number of world states
    # model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'deepseek/deepseek-r1', 'anthropic/claude-3.7-sonnet']
    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']

    experiment_name = "2025-04-20_23-07-21"
    log_file_path = f'./results/consistency_metric/Single_Round/{experiment_name}.txt'
    LLM_prediction_file_path = f'./results/LLM_predictions/Single_Round/{experiment_name}.json'

    with open(log_file_path, 'w') as f:
        # overall consistency metric
        phi = calculate_phi_single_round(LLM_prediction_file_path, model_list)
        number_of_agents = len(model_list)
        consistency_metric = calculate_consistency_metric(phi, number_of_agents, 2)

        '''
        print("------")
        print("joint posterior beliefs distribution: phi")
        for k in phi.keys():
            print(f"f: {np.frombuffer(k, dtype=np.float32).reshape(len(model_list), L)}")
            print(f"phi(f): {phi[k]}")
        print("------")
        '''
        
        print(f"Models: {model_list}")
        print(f"Consistency Metric: {consistency_metric}")

        for idx, model_name in enumerate(model_list):
            f.write(f"Agent {idx}: {model_name}\t")
        f.write('\n')
        f.write(f'Consistency Metric: {consistency_metric}\n')

        # pairwise consistency metric
        for agent1 in model_list:
            for agent2 in model_list:
                agents = [agent1, agent2]
                phi_pair = calculate_phi_single_round(LLM_prediction_file_path, agents)
                number_of_agents_pair = len(agents)
                consistency_metric_pair = calculate_consistency_metric(phi_pair, len(agents), 2)

                '''
                print("------")
                print("joint posterior beliefs distribution: phi")
                for k in phi_pair.keys():
                    print(f"f: {np.frombuffer(k, dtype=np.float32).reshape(len(agents), L)}")
                    print(f"phi(f): {phi_pair[k]}")
                print("------")
                '''
                
                print(f"Models: {agents}")
                print(f"Consistency Metric: {consistency_metric_pair}")

                for idx, model_name in enumerate(agents):
                    f.write(f"Agent {idx}: {model_name}\t")
                f.write('\n')
                f.write(f'Consistency Metric: {consistency_metric_pair}\n')
