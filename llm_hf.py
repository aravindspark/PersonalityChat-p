from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import torch
from transformers import pipeline
from huggingface_hub import login
from includes import model_resp_fmt, model_resp_scale

def login_to_hf_hub(token=""):
    login(token)

class ModelLlma:
    def __init__(self, model_name="Llama-3.2-3B-Instruct", load_quantized=False):
        llama_id = f"meta-llama/{model_name}"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {llama_id} from Huggingface!")
        if load_quantized:
            bnb_config = BitsAndBytesConfig(load_in_4bit= True,
                                            bnb_4bit_quant_type="nf4",
                                            )
            self.model = AutoModelForCausalLM.from_pretrained(llama_id,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="auto",
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(llama_id,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="auto")

        self.tokenizer = AutoTokenizer.from_pretrained(llama_id, use_fast=False)
        # self.history = []
        
    def get_response(self, statements, response_format="number"):

        prompt_scale = model_resp_scale.get(response_format, "")
        prompt_format = model_resp_fmt.get(response_format, "")
        
        system_prompt = f"""You are a helpful assistant who answers the questions in following format,
                          Question number: Your Answer
                          You can only reply from the following scale 
                          {prompt_scale}.
                          reply only above scale in the given statements."""
                          
        user_prompt = f"Here are a number of characteristics that may or may not apply to you. \
                        Please indicate the extent to which you agree or disagree with that statement. \
                        {prompt_scale} \
                        Here are the statements, score them one by one: \n {statements} \n \
                        Please reply in the format: \"statement index: score\". reply only the {prompt_format}. \
                        "
        
        prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]


        prompt_template = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True
        )
        
        input_tokens = self.tokenizer(prompt_template, return_tensors="pt").to(self.device)

        # set a pad token to avoid warnings with Llama2 like

        # Generate response
        output_tokens = self.model.generate(
            **input_tokens,
            do_sample=False,
            temperature=None, # unset to avoid warnings
            top_p=None, # unset to avoid warnings
            max_new_tokens=1000,
        )

        response_tokens = output_tokens[0][input_tokens["input_ids"].shape[-1]:] 
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        # Update history
        # self.history.append({"role": "user", "content": user_prompt})
        # self.history.append({"role": "assistant", "content": response})
        
        return response

class ModelQwen:
    def __init__(self, model_name="Qwen/Qwen3-4B", load_quantized=False):
        
        qwen_id = f"meta-llama/{model_name}"

        if load_quantized:
            bnb_config = BitsAndBytesConfig(load_in_4bit= True,
                                            bnb_4bit_quant_type="nf4",)
            self.model = AutoModelForCausalLM.from_pretrained(qwen_id,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="auto",
                                                          quantization_config=bnb_config,
                                                          trust_remote_code=True)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(qwen_id,
                                                          torch_dtype=torch.bfloat16,
                                                          device_map="auto")

        self.tokenizer = AutoTokenizer.from_pretrained(qwen_id)
        self.history = []

    def get_response(self, statements, response_format="numbers"):
        
        prompt_scale = model_resp_scale.get(response_format, "")
        prompt_format = model_resp_fmt.get(response_format, "")
        
        system_prompt = f"""Please answer questions from a human perspective and do not mention that you are an AI, Answer the questions in following format,
                          Question number: Your Answer
                          You can only reply from the following scale 
                          {prompt_scale}.
                          reply only above scale in the given statements."""
                          
        user_prompt = f"You can only reply {prompt_format} in the following statements. \
                        Here are a number of characteristics that may or may not apply to you. \
                        Please indicate the extent to which you agree or disagree with that statement. \
                        {prompt_scale} \
                        Here are the statements, score them one by one: \n {statements} \n \
                        Please reply in the format: \"statement index: score\". reply only the {prompt_scale}. \
                        "
    
        messages = self.history + [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        response_ids = self.model.generate(**inputs, max_new_tokens=1000)[0][len(inputs.input_ids[0]):].tolist()
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        # Update history
        # self.history.append({"role": "user", "content": user_input})
        # self.history.append({"role": "assistant", "content": response})

        return response
