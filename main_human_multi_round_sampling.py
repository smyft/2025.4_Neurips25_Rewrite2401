import json
import numpy as np
import os

from calculating_metrics.consistency_metric import calculate_consistency_metric

def calculate_phi_with_human_multi_round_sampling(data_file_path, prediction_file_path, model_list):
    with open(prediction_file_path, 'r') as f:
        LLM_data = json.load(f)
        LLM_data = LLM_data[0]["predictions"]

    with open(data_file_path, 'r') as f:
        human_data = json.load(f)
        human_data = human_data[0]["financial_news"]

    results = []
    for LLM_date_dict, human_date_dict in zip(LLM_data, human_data):
        LLM_date = LLM_date_dict["date"]
        human_date = human_date_dict["date"]
        assert LLM_date == human_date

        sample_predictions = LLM_date_dict["predictions"]
        joint_posterior_list = []

        date_data = human_date_dict["news"]

        for sample_dict in sample_predictions:
            error_data = False
            joint_posterior = []

            sample_idx = sample_dict["Sample"]
            round_predictions = sample_dict["predictions"][0]
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

                    headline = prediction_dict["headline"]
                    content = prediction_dict["content"]

                    for data_dict in date_data:
                        if data_dict["title"] == headline and data_dict["content"] == content:
                            human_sentiment = data_dict["sentiment"]
                            neg = float(human_sentiment["neg"])
                            neu = float(human_sentiment["neu"])
                            pos = float(human_sentiment["pos"])
                            human_posterior = np.array([pos + neu / 2, neg + neu / 2], dtype=np.float32)
                            joint_posterior.append(human_posterior)

            if not error_data:
                joint_posterior = np.stack(joint_posterior, axis=0)
                joint_posterior_list.append(joint_posterior)

        phi = {}
        number_of_signals = len(joint_posterior_list)
        for joint_posterior in joint_posterior_list:
            phi[joint_posterior.tobytes()] = phi.get(joint_posterior.tobytes(), 0) + 1/number_of_signals

        results.append({"date": LLM_date, "phi": phi})

    return results

if __name__ == "__main__":
    L = 2  # number of world states
    # model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'deepseek/deepseek-r1', 'anthropic/claude-3.7-sonnet']
    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']
    
    experiment_name = "2025-05-05_17-14-05"
    log_file_path = f'./results/consistency_metric/Human_Multi_Round_Sampling/{experiment_name}.txt'
    LLM_prediction_file_path = f'./results/LLM_predictions/Multi_Round_Sampling/{experiment_name}.json'
    data_file_path = "./data/financial_news_headlines/origin_data_2025-01-01_2025-01-31_AAPL_split.json"

    with open(log_file_path, 'w') as f:
        for agent in model_list:
            agents = [agent]
            results = calculate_phi_with_human_multi_round_sampling(data_file_path, LLM_prediction_file_path, agents)

            f.write('\n')
            for idx, model_name in enumerate(agents):
                f.write(f"Agent {idx}: {model_name}\t")
            f.write('\n')

            for date_dict in results:
                date = date_dict["date"]
                phi = date_dict["phi"]

                print(f"\nDate: {date}")
                f.write(f"\nDate: {date}\n")

                if phi == {}:
                    consistency_metric = None
                else:
                    consistency_metric = calculate_consistency_metric(phi, 2, 2)

                print(f"Consistency Metric: {consistency_metric}")
                f.write(f'Consistency Metric: {consistency_metric}\n')
    