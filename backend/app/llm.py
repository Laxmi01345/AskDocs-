import os
import re
import time
import logging
from dotenv import load_dotenv
from groq import Groq
from langchain_core.language_models.llms import LLM

load_dotenv()

logger = logging.getLogger(__name__)

_groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

_cerebras_client = None
try:
    from cerebras.cloud.sdk import Cerebras
    _cerebras_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))
except Exception:
    pass

GROQ_MODELS = ["openai/gpt-oss-20b", "allam-2-7b", "qwen/qwen3.6-27b", "openai/gpt-oss-120b"]


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
    return text.strip()


class GroqLLM(LLM):
    max_tokens: int = 300

    def _call(self, prompt: str, stop=None) -> str:
        last_error = None

        for attempt in range(len(GROQ_MODELS) * 2):
            model = GROQ_MODELS[attempt % len(GROQ_MODELS)]
            try:
                chat_completion = _groq_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )
                content = chat_completion.choices[0].message.content or ""
                if model.startswith("qwen/"):
                    content = _strip_thinking(content)
                return content
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str:
                    logger.warning(f"Rate limited on {model}, switching...")
                    time.sleep(1)
                    continue
                break

        if _cerebras_client:
            try:
                chat_completion = _cerebras_client.chat.completions.create(
                    model="gpt-oss-120b",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=self.max_tokens,
                )
                return chat_completion.choices[0].message.content
            except Exception:
                pass

        raise last_error

    @property
    def _identifying_params(self):
        return {"name": "groq-llm", "max_tokens": self.max_tokens}

    @property
    def _llm_type(self) -> str:
        return "groq-llm"


CerebrasLLM = GroqLLM
