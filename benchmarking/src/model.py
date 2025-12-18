import os
import re
import time
import torch
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import load_yaml, get_response_format, parse_json_response, get_model_name

load_dotenv("./.env")


class LLM:
    def __init__(
        self,
        model_type="openai",
        model_path="gpt-4o",
        num_retries=5,
        device=-1,
        use_cot=False,
        eval_only=False,
        is_llama=False
    ):
        self.model_type = model_type
        self.model_path = model_path
        self.model_name = get_model_name(model_path)
        self.num_retries = num_retries
        self.device = device if device >= 0 else "cpu"
        self.use_cot = use_cot
        self.is_llama = is_llama
        self.prompts = load_yaml("src/configs/prompts.yaml")

        if not eval_only:
            self.load_model()

    def init_prompt(self, task, lang):
        self.task = task
        self.lang = lang
        sys_prompt = self.prompts["sys"][lang]
        output_format = get_response_format(task, lang, use_cot=self.use_cot)
        sys_prompt += output_format
        if self.model_type == "HF" and "gemma" in self.model_name.lower():
            self.messages = []
            self.gemma_sys_prompt = sys_prompt
            return
        if "qwen" in self.model_name.lower() and not self.use_cot:
            sys_prompt = "/no_think\n\n" + sys_prompt
        self.messages = [{"role": "system", "content": sys_prompt}]

    def load_model(self):
        if self.model_type == "openai":
            self.client = ChatOpenAI(
                model=self.model_path,
                api_key=os.environ.get("API_KEY"),
                temperature=0.6,
            )
        elif self.model_type == "openai-compatible":
            self.client = ChatOpenAI(
                model=self.model_path,
                base_url=os.environ.get("API_URL"),
                api_key=os.environ.get("API_KEY"),
                temperature=0.6,
            )
        elif self.model_type == "HF":
            self.client = pipeline(
                "text-generation", model=self.model_path, device=self.device
            )
        else:
            print("Model is not supported at the moment")
            self.client = None

        print(f"> {self.model_name} loaded successfully")

    def _format_hf_prompt(self, msg):
        if "llama" in self.model_name.lower():
            sys = self.messages[0]["content"]
            return (
                f"<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n"
                f"{msg}\n[/INST]"
            )

        if "gemma" in self.model_name.lower():
            return (
                "<start_of_turn>system\n"
                f"{self.gemma_sys_prompt}\n"
                "<end_of_turn>\n"
                "<start_of_turn>user\n"
                f"{msg}\n"
                "<end_of_turn>\n"
                "<start_of_turn>model\n"
            )

        return msg


    def gen_response(self, msg):
        fallback = (
            {"answer": ""} if self.task == "EA" else
            {"answer_q1": "", "answer_q2": ""}
        )

        for attempt in range(self.num_retries):

            try:
                # HF MODE
                if self.model_type == "HF":
                    prompt = self._format_hf_prompt(msg)

                    out = self.client(
                        prompt,
                        max_new_tokens=512 if self.use_cot else 50,
                        return_full_text=False,
                    )[0]["generated_text"]
                    #print(out)
                # OPENAI MODE
                else:
                    messages = self.messages + [{"role": "user", "content": msg}]
                    out = self.client.invoke(messages).content

                # PARSE JSON
                parsed = parse_json_response(out)
                if parsed is None:
                    raise ValueError("bad json")

                return parsed

            except Exception:
                if attempt < self.num_retries - 1:
                    msg = "Return ONLY VALID JSON. Try again."
                    continue
                return fallback