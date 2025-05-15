import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import scipy
import matplotlib

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

def get_human_consistency(human_consistency_path):
    data = {}
    with open(human_consistency_path, 'r') as f:
        lines = f.readlines()

        for line in lines:
            line = line.strip()
            if 'Consistency Metric' in line:
                consistency_metric = float(line.split(' ')[-1])
                data[agent] = consistency_metric
            else:
                agent = line.split(" ")[-1]

    return data

def get_accurany():
    data = {
        'deepseek/deepseek-chat-v3-0324': 0.82,
        'meta-llama/llama-4-scout': 0.75,
        'google/gemini-2.0-flash-001': 0.72,
        'meta-llama/llama-3.3-70b-instruct': 0.71,
        'openai/gpt-4o-mini': 0.65,
        'qwen/qwen-turbo': 0.63,
        'google/gemini-flash-1.5': 0.57
    }
    return data

name_map = {
    'openai/gpt-4o-mini': "GPT-4o mini",
    'google/gemini-2.0-flash-001': "Gemini 2.0 Flash",
    'google/gemini-flash-1.5': "Gemini 1.5 Flash",
    'deepseek/deepseek-chat-v3-0324': "Deepseek V3",
    'meta-llama/llama-4-scout': "Llama 4 Scout",
    'meta-llama/llama-3.3-70b-instruct': "Llama 3 70B Instruct",
    'qwen/qwen-turbo': "Qwen Turbo"
}

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
    experiment_name = "2025-04-21_18-44-41"
    experiment_name = "2025-04-28_17-17-45"
    human_consistency_path_file = f'./results/consistency_metric/Human_Multi_Round/{experiment_name}.txt'

    model_list = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']
    
    consistency_dict = get_human_consistency(human_consistency_path_file)
    accuracy_dict = get_accurany()

    consistency_list = []
    accuracy_list = []
    point_labels = []
    for agent in model_list:
        consistency_list.append(consistency_dict[agent])
        accuracy_list.append(accuracy_dict[agent])
        point_labels.append(agent)

    print(matplotlib.get_cachedir())
    plt.rcParams['font.family'] = 'Calibri'

    plt.figure(figsize=(12, 10))
    k, b = np.polyfit(consistency_list, accuracy_list, 1)
    x_line = np.linspace(min(consistency_list), max(consistency_list), 100)
    y_line = k * x_line + b
    plt.plot(x_line, y_line, color='red', label=f'Linear Fit')
    plt.scatter(consistency_list, accuracy_list, label='Data Point', color='blue')

    x_range = max(consistency_list) - min(consistency_list)
    y_range = max(accuracy_list) - min(accuracy_list)
    x_offset = x_range * 0.005
    y_offset = y_range * 0.01
    for x, y, label in zip(consistency_list, accuracy_list, point_labels):
        # plt.text(x, y, label) # Default: text starts exactly at (x,y)
        # Adding offset to place text slightly to the top-right
        plt.text(x + x_offset, y + y_offset, name_map[label], fontsize=10, ha='left', va='bottom')

    plt.xlabel("Consistency Metric")
    plt.ylabel("Accuracy on MMLU-Pro")
    # plt.title('Scatter Plot with Linear Regression')
    output_file_path = f'./results/figures/Human_Multi_Round/{experiment_name}.pdf'
    plt.savefig(output_file_path, bbox_inches='tight')
    tikz_path = f'./results/tikz/Human_Multi_Round/{experiment_name}.tikz'
    tikzplotlib.save(tikz_path)
    plt.close()
    
    Y = pd.Series(accuracy_list, name="Persistent Disagreement")
    X = pd.Series(consistency_list, name="Consistency Metric")
    run_regression(X, Y)
