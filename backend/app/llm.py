import os
from dotenv import load_dotenv
from cerebras.cloud.sdk import Cerebras
from langchain_core.language_models.llms import LLM

load_dotenv()

_client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))


class CerebrasLLM(LLM):
    max_tokens: int = 300

    def _call(self, prompt: str, stop=None) -> str:
        chat_completion = _client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=self.max_tokens,
        )
        return chat_completion.choices[0].message.content

    @property
    def _identifying_params(self):
        return {"name": "cerebras-llm", "max_tokens": self.max_tokens}

    @property
    def _llm_type(self) -> str:
        return "cerebras-llm"
