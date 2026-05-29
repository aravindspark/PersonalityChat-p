from openai import OpenAI
from openai import APIConnectionError
import time
from includes import model_resp_fmt, model_resp_scale 

api_models_list = [
    "llama-3-8b-instruct",
    "llama-3.3-70b-instruct",
    "qwen-2.5-7b-instruct", 
    "qwen-2.5-72b-instruct",
    "deepseek-r1-distil-qwen-14b",
]

api_base_url = "https://api.openai.com/v1" # Replace with your API base URL

class OpenAIClient:
    def __init__(self, api_key, model_name="llama-3-8b-instruct"):
        self.client = OpenAI(base_url=api_base_url, api_key=api_key)
        self.model_name = model_name
     
    def get_response(self, statements, response_format="number"):
        
        system_prompt = f"""You are a helpful assistant who can only reply {model_resp_fmt.get(response_format, "")} in every statement.
        All reponse must follow the following format,\n \"statement index: score\"."""

        user_prompt = f"""Here are a number of characteristics that may or may not apply to you.
        Please indicate the extent to which you agree or disagree with the statement.
        \n{model_resp_scale.get(response_format,"")}\n
        Here are the statements, score them one by one: \n {statements} \n
        Please reply in the format: \"statement index: score\".
        reply only the {model_resp_fmt.get(response_format,"")}."""

        # print(f"System Prompt:\n {system_prompt}  \nUser Prompt:\n {user_prompt}")
        try:
            response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        )
        except APIConnectionError as e:
            print(f"API Connection Error: {e}")
            # Retry in 2 seconds
            # time.sleep(2)
            return None
        
        except Exception as e:
            print(f"API Error: failed to get response from {self.model_name}: {e}")
            return None
        
        return response.choices[0].message.content
