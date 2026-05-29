
# PersonalityChat -  Personality trait Test on LLMs using BFI Inventory

## Overview
This project the Big Five Inventory (BFI) personality test aims to study the Personality profile of Opensource and Closed source Large Language Models (LLMs). The source supports inference with LLMs via OpenAI client and HuggingFace transformers models, this can be extended. 
The project focuses on BFI questionnaire as it is a well established personality trait test method. 
The tool collects and processes responses from the LLM, generates summary statistics and visualizations.

## Features
- Supports HuggingFace transformer models
- Supports OpenAI API client
- Configurable test parameters via config file

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib
- HuggingFace transformers and access token for HF transformer models
- Access to openai tokens for openai API client

## Setup
1. Clone the repository to a local directory.
2. Install dependencies:
	 ```sh
	 pip install -r requirements.txt
	 ```
3. Prepare your configuration file (check `test_config.json` to start).

## How to Run ?
Run the test script with your configuration:

```sh
python run_test.py --config configs/config_qwen3-14B-Base.json
```

- `--config`: Path to the configuration JSON file.
- `--iterations`: Number of times to run the test (default: 10). overrides iteration from config

## Configuration File Format E.g., (`test_config.json`)
```json
{
	"model-client": "hf-transformers",  				// or ["openapi"] - How to do inference with the model 
	"model-name": "qwen2.5-7b-Instruct",	// Model name to load , any valid model name
	"iterations": 10,						// Number of iterations to run the test
	"load_quantized": false,				// Load the model in quantized form 	
	"bfi_test_type": "category",  			//  or ["individual"] - category wise or Individual wise test
	"response_format": "number",  	// or ["label", "text"]  model reply scale
	"is_shuffle": true 				//	Whether to randomly shuffle the questions from the orginal format
}
```

## Test Outputs
Results and logs from the test are captured in a directory named after the model.
- `bfi_responses.csv`: All the LLM responses captured along with the questions.
- `{model}_results.md`: Summary of the results.
- `{model}_chart.png`: Polar chart of average scores.

## Customization
- Questionnaires and config file can be modified as per user needs.
- The software can be extended to support more models or response formats.