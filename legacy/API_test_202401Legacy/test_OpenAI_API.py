import json
from openai import OpenAI
import openai


def interact_with_GPT(model, system_messages, user_messages, json_file_path, temperature=0, logprobs=True, seed=42, max_tokens=1000, top_logprobs=2):
    client = OpenAI()

    if logprobs:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_messages}
                ],
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs
            )
        except openai.error.InvalidRequestError as e:
            print(f"Invalid request error: {e}")
        except openai.error.AuthenticationError as e:
            print(f"Authentication error: {e}")
        except openai.error.APIError as e:
            print(f"API error: {e}")
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

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

    else:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_messages},
                    {"role": "user", "content": user_messages}
                ],
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                logprobs=False
            )
        except openai.error.InvalidRequestError as e:
            print(f"Invalid request error: {e}")
        except openai.error.AuthenticationError as e:
            print(f"Authentication error: {e}")
        except openai.error.APIError as e:
            print(f"API error: {e}")
        except openai.error.RateLimitError as e:
            print(f"Rate limit error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    content_json_format = {"system": system_messages, "user": user_messages, "assistant": completion.choices[0].message.content}
    if logprobs:
        content_json_format["logprobs"] = logprob_json_format

    with open(json_file_path, "w") as json_file:
        json.dump(content_json_format, json_file)

    print(f"Model:{model}, Logprob: {logprobs}")
    print(f"Response:{completion.choices[0].message.content}")
    print("\n")

    return content_json_format


def interact_with_GPT_with_history(model, dialogue_history, log_file_path, temperature=0, logprobs=True, seed=42, max_tokens=1000, top_logprobs=2):
    client = OpenAI()

    if logprobs:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=dialogue_history,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=top_logprobs
            )
        except openai.BadRequestError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned a bad request Error: {e}")
        except openai.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")

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

    else:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=dialogue_history,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
                logprobs=False
            )
        except openai.BadRequestError as e:
            #Handle API error here, e.g. retry or log
            print(f"OpenAI API returned a bad request Error: {e}")
            pass
        except openai.APIConnectionError as e:
            #Handle connection error here
            print(f"Failed to connect to OpenAI API: {e}")
            pass
        except openai.RateLimitError as e:
            #Handle rate limit error (we recommend using exponential backoff)
            print(f"OpenAI API request exceeded rate limit: {e}")
            pass

    log_json_format = {"dialogue_history": dialogue_history.copy()}
    log_json_format["dialogue_history"].append({"role": "assistant", "content": completion.choices[0].message.content})
    if logprobs:
        log_json_format["logprobs"] = logprob_json_format

    if log_file_path is not None:
        with open(log_file_path, "w") as log_file:
            json.dump(log_json_format, log_file)

    print(f"Model:{model}, Logprob: {logprobs}")
    print(f"Response:{completion.choices[0].message.content}")
    print("\n")

    return log_json_format


if __name__ == "__main__":
    models = ['gpt-4o-2024-05-13', 'gpt-4o-mini-2024-07-18', 'gpt-4-turbo-2024-04-09', 'gpt-4-0613', 'gpt-3.5-turbo-0125']
    system_message = "You are a poetic assistant."
    user_message = "Hello"
    dialogue_history = [{"role": "system", "content": system_message}, {"role": "user", "content": user_message}]

    '''
    for model in models:
      interact_with_GPT(model, system_message, user_message, f"./API_test/logprob_{model}.json", logprobs=True)
      interact_with_GPT(model, system_message, user_message, f"./API_test/{model}.json", logprobs=False)
    '''

    for model in models:
        interact_with_GPT_with_history(model, dialogue_history, f"./API_test/logprob_{model}_history.json", logprobs=True)
        interact_with_GPT_with_history(model, dialogue_history, f"./API_test/{model}_history.json", logprobs=False)
