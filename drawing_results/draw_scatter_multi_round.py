import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy

import tikzplotlib
import statsmodels.api as sm
import pandas as pd

def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)

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


def run_regression(X, Y):
    X_with_constant = sm.add_constant(X)
    model = sm.OLS(Y, X_with_constant)
    results = model.fit()
    coefficients = results.params
    r_squared = results.rsquared
    adjusted_r_squared = results.rsquared_adj

    print("Regression Results:")
    print("===================")
    print(f"Independent Variable(s): {X.name}")
    print(f"Dependent Variable: {Y.name}")
    print("-------------------")

    print("Coefficient Summary:")
    standard_errors = results.bse
    p_values = results.pvalues
    for term in coefficients.index:
        print(f"  {term.capitalize():<10} | Coef: {coefficients[term]:>9.4f} | Std Err: {standard_errors[term]:>9.4f} | P>|t|: {p_values[term]:>9.4f}")

    
    print("Coefficients:")
    print(f"  Intercept: {coefficients.iloc[0]:.4f}") # Assuming the constant is the first param
    print(f"  X (slope): {coefficients.iloc[1]:.4f}") # Assuming X is the second param
    print("-------------------")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {adjusted_r_squared:.4f}")
    print("===================\n")


if __name__ == "__main__":
    experiment_name = "2025-04-28_17-17-45"
    result_file_path = f'./results/consistency_metric/Multi_Round/{experiment_name}.txt'
    LLM_prediction_file_path = f'./results/LLM_predictions/Multi_Round/{experiment_name}.json'

    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']
    
    consistency_metric_data = parse_consistency_files(result_file_path)

    consistency_metric_list = []
    js_divergence_last_list = []
    tv_distance_last_list = []
    js_divergence_first_list = []
    tv_distance_first_list = []

    for agent1 in model_list:
        for agent2 in model_list:
            if agent1 == agent2:
                continue
            agents = [agent1, agent2]
            number_of_agents_pair = len(agents)
            phi_pair_list = calculate_phi_multi_round(LLM_prediction_file_path, agents)

            consistency_metric = consistency_metric_data[(agent1, agent2)][0]
            consistency_metric_list.append(consistency_metric)
            
            phi_last = phi_pair_list[-1]
            js_divergence = 0
            tv_distance = 0
            for k in phi_last.keys():
                joint_posterior = np.frombuffer(k, dtype=np.float32).reshape(2, 2)
                # JS divergence
                js = scipy.spatial.distance.jensenshannon(joint_posterior[0], joint_posterior[1])
                tv = 0.5 * np.sum(np.abs(joint_posterior[0] - joint_posterior[1]))

                js_divergence += phi_last[k] * js
                tv_distance += phi_last[k] * tv
            js_divergence_last_list.append(js_divergence)
            tv_distance_last_list.append(tv_distance)

            phi_first = phi_pair_list[0]
            js_divergence = 0
            tv_distance = 0
            for k in phi_first.keys():
                joint_posterior = np.frombuffer(k, dtype=np.float32).reshape(2, 2)
                # JS divergence
                js = scipy.spatial.distance.jensenshannon(joint_posterior[0], joint_posterior[1])
                tv = 0.5 * np.sum(np.abs(joint_posterior[0] - joint_posterior[1]))

                js_divergence += phi_first[k] * js
                tv_distance += phi_first[k] * tv
            js_divergence_first_list.append(js_divergence)
            tv_distance_first_list.append(tv_distance)


    consistency_metric_list = np.array(consistency_metric_list)
    js_divergence_last_list = np.array(js_divergence_last_list)
    tv_distanc_last_list = np.array(tv_distance_last_list)
    js_divergence_first_list = np.array(js_divergence_first_list)
    tv_distanc_first_list = np.array(tv_distance_first_list)

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(consistency_metric_list, js_divergence_last_list, 1)
    x_line = np.linspace(min(consistency_metric_list), max(consistency_metric_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(consistency_metric_list, js_divergence_last_list, label='Data Point', color='blue')
    plt.xlabel("Consistency Metric")
    plt.ylabel("Persistent Disagreement")
    # plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round/{experiment_name}_consistency_persistent.pdf'
    plt.savefig(output_file_path, bbox_inches='tight')
    tikz_path = f'./results/tikz/Multi_Round/{experiment_name}_consistency_persistent.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()    
    
    Y = pd.Series(js_divergence_last_list, name="Persistent Disagreement")
    X = pd.Series(consistency_metric_list, name="Consistency Metric")
    run_regression(X, Y)
    

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(consistency_metric_list, js_divergence_first_list, 1)
    x_line = np.linspace(min(consistency_metric_list), max(consistency_metric_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(consistency_metric_list, js_divergence_first_list, label='Data Point', color='blue')
    plt.xlabel("Consistency Metric")
    plt.ylabel("Initial Disagreement")
    # plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round/{experiment_name}_consistency_initial.pdf'
    plt.savefig(output_file_path, bbox_inches='tight')
    tikz_path = f'./results/tikz/Multi_Round/{experiment_name}_consistency_initial.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()
    
    Y = pd.Series(js_divergence_first_list, name="Initial Disagreement")
    X = pd.Series(consistency_metric_list, name="Consistency Metric")
    run_regression(X, Y)

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(js_divergence_first_list, js_divergence_last_list, 1)
    x_line = np.linspace(min(js_divergence_first_list), max(js_divergence_first_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(js_divergence_first_list, js_divergence_last_list, label='Data Point', color='blue')
    plt.xlabel("Initial Disagreement")
    plt.ylabel("Persistent Disagreement")
    # plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round/{experiment_name}_initial_persistent.pdf'
    plt.savefig(output_file_path, bbox_inches='tight')
    tikz_path = f'./results/tikz/Multi_Round/{experiment_name}_initial_persistent.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()
    
    Y = pd.Series(js_divergence_last_list, name="Persistent Disagreement")
    X = pd.Series(js_divergence_first_list, name="Initial Disagreement")
    run_regression(X, Y)

    '''

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(consistency_metric_list, tv_distance_last_list, 1)
    x_line = np.linspace(min(consistency_metric_list), max(consistency_metric_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(consistency_metric_list, tv_distance_last_list, label='Data Point', color='blue')
    plt.xlabel("Consistency Metric")
    plt.ylabel("TV Distance")
    # plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round/{experiment_name}_TVDistance.pdf'
    plt.savefig(output_file_path, bbox_inches='tight')

    tikz_path = f'./results/tikz/Multi_Round/{experiment_name}_TVDistance.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(consistency_metric_list, tv_distance_first_list, 1)
    x_line = np.linspace(min(consistency_metric_list), max(consistency_metric_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(consistency_metric_list, tv_distance_first_list, label='Data Point', color='blue')
    plt.xlabel("Consistency Metric")
    plt.ylabel("TV Distance")
    # plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round/{experiment_name}_TVDistance_Round0.pdf'
    plt.savefig(output_file_path, bbox_inches='tight')

    tikz_path = f'./results/tikz/Multi_Round/{experiment_name}_TVDistance_Round0.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(tv_distance_first_list, tv_distance_last_list, 1)
    x_line = np.linspace(min(tv_distance_last_list), max(tv_distance_last_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(tv_distance_first_list, tv_distance_last_list, label='Data Point', color='blue')
    plt.xlabel("Initial TV Distance")
    plt.ylabel("TV Distance")
    # plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Multi_Round/{experiment_name}_TVDistance_TVDistance.pdf'
    plt.savefig(output_file_path, bbox_inches='tight')

    tikz_path = f'./results/tikz/Multi_Round/{experiment_name}_TVDistance_TVDistance.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()
    '''
