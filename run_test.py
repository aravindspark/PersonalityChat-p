from llm_api import OpenAIClient, api_models_list
from llm_hf import *
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import warnings
import re

result_dir = 'results'
bfi_pre_text = "I see Myself as Someone Who"

reverse_score = { 1: 5, 2: 4, 3: 3, 4: 2, 5: 1 }

label_to_score = { 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5 }

text_to_score = {
    'disagree strongly': 1,
    'disagree a little': 2,
    'neither agree nor disagree': 3,
    'agree a little': 4,
    'agree strongly': 5
}

category_names = {
    'E': "Extraversion",
    'A': "Agreeableness",
    'C': "Conscientiousness",
    'N': "Neuroticism",
    'O': "Openness"
}

bfi_test_type = [ "category", "individual"]
response_format = [ "number", "label", "text" ]

def draw_polar_chart(data_frame, title):
    llm_scores = data_frame.get('avg_score')
    categories = list(llm_scores.keys())
    values = list(llm_scores.values())
    xticks_labels = [f"{label}\n({val})" for label, val in zip(categories, values)]
    
    values += values[:1]
    
    # compute angle for each category
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 
    
    #draw polar chart
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=1, linestyle='solid',color='orangered')
    ax.fill(angles, values, color='orangered', alpha=0.10)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(xticks_labels,  size=10)
    
    ax.set_ylim(1,5)
    ax.set_rgrids([1, 2, 3, 4, 5], labels=[])
    ax.grid(True)

    plt.title(title.replace('_',' '), size=14, pad=50)
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{title}_chart.png", dpi=300, bbox_inches='tight')

def process_response(response, bfi_data=None, resp_ncol=0, resp_fmt=""):
    for line in response.splitlines():
        line = line.strip()
        # print(f"Processing line: {line}")
        if not line:
            continue
        match1 = re.match(r'^(\d{1,2})[\.:]\s*?(.*)$', line)
        if match1 is not None:
            q_num = int(match1.group(1))
            resp_txt = match1.group(2).strip()
            # check for addinal text in response text
            match2 = re.match(r'^.*[\.:](.*)$', resp_txt)
            if match2 is not None:
                resp_txt = match2.group(1).strip()
        else:
            continue

        # print(f"{q_num} : {resp_txt}")
        try:
            if resp_txt == "":
                resp_val = 0
            else:
                if resp_fmt == "number":
                    resp_val = int(resp_txt)
                elif resp_fmt == "label":
                    resp_val = label_to_score.get(resp_txt.lower(), 0)
                elif resp_fmt == "text":
                    resp_val = text_to_score.get(resp_txt.lower(), 0)

            # print(f"Question Number: {q_num}, Response Value: {resp_val}")
            # Update the DataFrame with the response
            if bfi_data.loc[bfi_data["number"] == q_num, "reversed"].values[0]:
                resp_val = reverse_score.get(resp_val, 0)
            bfi_data.loc[bfi_data["number"] == q_num, f"response_{resp_ncol}"] = resp_val
        except ValueError:
            continue
        
def get_openapi_model(model_name):
    if model_name not in api_models_list:
        print(f"Selected Model | {model_name} | is not supported. \
                Supported models: {api_models_list}")
        exit(1)
    return OpenAIClient(api_key=os.getenv("LLM_API_KEY"), model_name=model_name)

def get_hf_model(model_name, load_quantized=False):
    if load_quantized:
        warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast.*")
    if "Qwen" in model_name:
        model = ModelQwen(model_name=model_name, load_quantized=load_quantized)
    if "Llama" in model_name:
        warnings.filterwarnings("ignore", message=".*Setting `pad_token_id` to `eos_token_id.*")
        model = ModelLlma(model_name=model_name, load_quantized=load_quantized)
    return model
        
def parse_cli_args():
    parser = argparse.ArgumentParser(description="Run Personality test with LLM")
    parser.add_argument("--config", type=str, default="test_config.json", help="Configuration file path")
    parser.add_argument("--iterations", type=int, default=0, help="Number of iterations for the test")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle Questions")
    return parser.parse_args()

def get_response_by_category(bfi_data, model, log_file , resp_fmt):
    responses = ""
    for category, group in bfi_data.groupby("category"):
        statements = ""
        for _, row in group.iterrows():
            questions = f"{row['number']}. {bfi_pre_text} {row['question']}\n"
            statements += questions
        model_response = model.get_response(statements, response_format=resp_fmt)
        responses += model_response + "\n"
        # print(f"model:\n{model_response}")
        # Log user input and model response
        log_file.write(f"Category: {category_names.get(category)}\n")
        log_file.write(f"Satements:\n{statements}\n")
        log_file.write(f"Model responses:\n{model_response}\n")
        log_file.write("-----------------------------\n")

    return responses

def get_response_individual(bfi_data, model, log_file, resp_fmt):
    responses = ""
    for _, row in bfi_data.iterrows():
        question = f"{row['number']}. {bfi_pre_text} {row['question']}\n"
        model_response = model.get_response(question, response_format=resp_fmt)
        responses += model_response + "\n"
        # print(f"model:\n{model_response}")
        # Log user input and model response
        log_file.write(f"Statement: {question}")
        log_file.write(f"Model response: {model_response}\n")

    log_file.write("-----------------------------\n")
    return responses

if __name__ == "__main__":
    
    print("\n----------------------------START-------------------------------\n")
    
    args = parse_cli_args()
    
    # Load configuration from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Get configuration values
    model_client = config.get("model-client", "openai")
    model_name = config.get("model-name", "qwen2.5-7b-Instruct")
    iterations = config.get("iterations", 10)
    load_quantized = config.get("load_quantized", False)
    test_type = config.get("bfi_test_type", "category")
    resp_fmt = config.get("response_format", "number")
    shuffle = config.get("is_shuffle", False)

    if args.iterations:
        iterations = args.iterations
    if args.shuffle:
        shuffle = args.shuffle

    if model_client == "openai":
        model = get_openapi_model(model_name)
    elif model_client == "hf-transformers":
        login_to_hf_hub(os.getenv('HUGGINGFACE_HUB_TOKEN',"Env Key Error!"))
        model = get_hf_model(model_name, load_quantized)
          
    # Read the questionnaire CSV
    bfi_data = pd.read_csv("questionaries_BFI.csv")
    bfi_data["number"] = bfi_data["number"].astype(int)
    
    # create directories for results
    test_root_dir = os.getcwd()
    # create results dir if not exits
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    os.chdir(result_dir)
    
    os.makedirs(model_name, exist_ok=True)
    os.chdir(model_name)

    if shuffle:
        os.makedirs(f"{test_type}/shuffled", exist_ok=True)
        os.chdir(f"{test_type}/shuffled")
    else:
        os.makedirs(test_type, exist_ok=True)
        os.chdir(test_type)
    
    print(f"Creating test results in {os.getcwd()}")
    
    
    # copy into a working copy
    bfi_response = bfi_data.copy()
    
    if shuffle:
        # Shuffle the questions
        bfi_response = bfi_response.sample(frac=1).reset_index(drop=True)

    for n in range(1,iterations+1):
        # Add a column for responses
        bfi_response[f"response_{n}"] = None
        print(f"Iteration {n} of {iterations}")
        # Open log file for writing
        with open(f"response{n}_log.txt", "w", encoding="utf-8") as log_file:
            log_file.write(f"model: {model_name}\n")
            responses_iter = ""

            if test_type=="category":
                log_file.write("Test type: Category\n\n")
                responses_iter = get_response_by_category(bfi_data, model, log_file, resp_fmt=resp_fmt)
            elif test_type=="individual":
                log_file.write("Test type: Individual\n\n")
                responses_iter = get_response_individual(bfi_data, model, log_file, resp_fmt=resp_fmt)
            else:
                print(f"Invalid test type: {test_type}. Supported types: {bfi_test_type}")
                exit(1)

            log_file.write("\n")

        # Parse responses and update DataFrame
        # print(responses_iter)
        process_response(responses_iter, bfi_response, resp_ncol=n, resp_fmt=resp_fmt)

    # check the processed response
    has_zeros = (bfi_response.iloc[:, 4:] == 0).any().any()

    if has_zeros:
        print("Warning: Process error 0 value in BFI responses.")

    # Save the DataFrame with responses
    bfi_response.to_csv(f"bfi_responses_{resp_fmt}.csv", index=False)

    # Merge all response columns by their average into a single column
    response_cols = [col for col in bfi_response.columns if col.startswith("response_")]
    bfi_response["response"] = bfi_response[response_cols].mean(axis=1)
    # bfi_response = bfi_response.drop(columns=response_cols)
    # Calculate and print average response for each category
    llm_scores = pd.DataFrame(columns=["Dimensions", "avg_score","std_score"])

    # print("\nAverage response for each category:")
    avg_by_category = bfi_response.groupby("category")["response"].mean()
    std_by_category = bfi_response.groupby("category")["response"].std()
    
    for category in avg_by_category.index:
        # add the average score to the table
        avg = avg_by_category[category]
        std = std_by_category[category]
        llm_scores.loc[len(llm_scores)] = [ category_names.get(category), round(avg,2), round(std,2) ]        
        # print(f"{category_names.get(category)}: \t\t{avg:.2f} ± {std:.2f}")
    
    llm_scores = llm_scores.set_index('Dimensions')
    
    print("---------------------------------------------------------------\n")
    print(llm_scores.to_markdown())
    print("\n----------------------------END--------------------------------\n")
    
    with open(f"results_{resp_fmt}.md", "w") as result_file:
        result_file.write(f"# Test information\n")
        result_file.write(f"- Model: **{model_name}**\n")
        result_file.write(f"- Test type: **{test_type}**\n")
        result_file.write(f"- Response format: **{resp_fmt}**\n")
        result_file.write(f"- Shuffle: **{shuffle}**\n")
        result_file.write(f"- No of iterations: **{iterations}**\n\n")
        result_file.write("# Model responses for each BFI dimensions:\n\n")
        result_file.write(llm_scores.to_markdown())
        result_file.write("\n\n")

    # Draw polar chart
    draw_polar_chart(llm_scores.to_dict(), f"{model_name}_{resp_fmt}")
    
    os.chdir(test_root_dir)
