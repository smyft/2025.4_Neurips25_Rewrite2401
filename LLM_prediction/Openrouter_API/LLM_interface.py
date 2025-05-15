import json
import os
import re
import sys

import numpy as np
from openai import OpenAI
from torch.utils.data import DataLoader
from tqdm import tqdm
import requests
from openai import OpenAI
import openai
import time

openrouter_key = "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"


def LLM_interface(model, dialogue_history, log_file_path, temperature=0, logprobs=True, seed=42, max_tokens=1000, top_logprobs=2, debug=False):
    # Load API key from environment variable
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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" # os.getenv("OPENROUTER_API_KEY"),
    )

    max_retries = 8
    base_delay = 1
    retryable_errors = (
        openai.APIConnectionError, 
        openai.RateLimitError,
        openai.APIError,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout
    )

    log_json_format = {
        "metadata": {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "seed": seed
        },
        "error_info": [],
        "dialogue_history": dialogue_history.copy()
    }

    retries = 0
    while True:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=dialogue_history,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                logprobs=False
            )
            try:
                finish_reason = completion.choices[0].finish_reason
                if finish_reason == "stop":
                    break
                else:
                    if retries >= max_retries:
                        with open(log_file_path, "w") as log_file:
                            json.dump(log_json_format, log_file, indent=2)
                        print(f"Request failed. Reach max retry: {max_retries}")
                        return None
            
                    log_json_format["error_info"].append({
                        "error_type": "CompletionError",
                        "error_message": f"Finish Reason: {finish_reason}",
                        "retries": retries
                    })
                    delay = base_delay * (2 ** retries)
                    print(f"Retry {retries+1}/{max_retries} in {delay}s: CompletionError - Finish Reason: {finish_reason}")
                    time.sleep(delay)
                    retries += 1
            except Exception as e:
                if retries >= max_retries:
                    with open(log_file_path, "w") as log_file:
                        json.dump(log_json_format, log_file, indent=2)
                    print(f"Request failed. Reach max retry: {max_retries}")
                    return None
                
                log_json_format["error_info"].append({
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "retries": retries
                })
                delay = base_delay * (2 ** retries)
                print(f"Retry {retries+1}/{max_retries} in {delay}s: {type(e).__name__} - {str(e)}")
                time.sleep(delay)
                retries += 1

        except retryable_errors as e:
            if retries >= max_retries:
                with open(log_file_path, "w") as log_file:
                    json.dump(log_json_format, log_file, indent=2)
                print(f"Request failed. Reach max retry: {max_retries}")
                return None
                
            log_json_format["error_info"].append({
                "error_type": type(e).__name__,
                "error_message": str(e),
                "retries": retries
            })
            delay = base_delay * (2 ** retries)
            print(f"Retry {retries+1}/{max_retries} in {delay}s: {type(e).__name__} - {str(e)}")
            time.sleep(delay)
            retries += 1

    log_json_format["dialogue_history"].append({
        "role": "assistant",
        "content": completion.choices[0].message.content
    })
    if logprobs:
        log_prob_content = completion.choices[0].logprobs.content
        logprob_json_format = []
        for token_dict in log_prob_content:
            dict = {}
            dict["token"] = token_dict.token
            # print(token_dict.token)
            dict["alternatives"] = []
            for token_substitutes_dict in token_dict.top_logprobs:
                dict["alternatives"].append({"token": token_substitutes_dict.token, "logprob": token_substitutes_dict.logprob})
                # print(token_substitutes_dict.token)
                # print(token_substitutes_dict.logprob)
            logprob_json_format.append(dict)
        log_json_format["logprobs"] = logprob_json_format

    with open(log_file_path, "w") as log_file:
        json.dump(log_json_format, log_file, indent=2)

    if debug:
        print(f"Model:{model}, Logprob: {logprobs}")
        print(f"Response:{completion.choices[0].message.content}")

    return log_json_format


if __name__ == "__main__":
    # example code
    '''
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=f"{openrouter_key}",
    )

    completion = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "Hello there"
            }
        ]
    )
    print(completion.choices[0].message.content)
    '''

    # models = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'deepseek/deepseek-r1', 'anthropic/claude-3.7-sonnet']
    
    models = ['openai/gpt-4o-mini', 'google/gemini-2.0-flash-001', 'google/gemini-flash-1.5', 'deepseek/deepseek-chat-v3-0324',
                  'meta-llama/llama-4-scout', 'meta-llama/llama-3.3-70b-instruct', 'qwen/qwen-turbo']

    user_message = "Hello"
    dialogue_history = [{"role": "user", "content": user_message}]

    for model in models:
        
        print("------")
        print(f"Model: {model}")

        model_name_for_path = model
        if '/' in model:
            model_name_for_path = "-".join(model.split('/'))
        
        # LLM_interface(model, dialogue_history, f"./LLM_prediction/Openrouter_API/test_results/logprob_{model_name_for_path}.json", logprobs=True, debug=True)
        LLM_interface(model, dialogue_history, f"./LLM_prediction/Openrouter_API/test_results/{model_name_for_path}.json", logprobs=False, debug=True)
        
        print("------")
