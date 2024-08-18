import os
import logging
from abc import ABC, abstractmethod

import instructor
from dotenv import load_dotenv
from instructor.client import T
from litellm import completion
from litellm.utils import validate_environment
from llama_index.core.base.llms.types import (
    CompletionResponse,
    CompletionResponseAsyncGen,
)
from llama_index.llms.litellm import LiteLLM

load_dotenv()


class BaseLLM(ABC):
    @abstractmethod
    async def astream(self, prompt: str) -> CompletionResponseAsyncGen:
        pass

    @abstractmethod
    def complete(self, prompt: str) -> CompletionResponse:
        pass

    @abstractmethod
    def structured_complete(self, response_model: type[T], prompt: str) -> T:
        pass


class EveryLLM(BaseLLM):
    def __init__(
        self,
        model: str,
    ):
        os.environ.setdefault("OLLAMA_API_BASE", "http://192.168.140.246:11434/api/generate")

        validation = validate_environment(model)
        if validation["missing_keys"]:
            raise ValueError(f"Missing keys: {validation['missing_keys']}")

        self.llm = LiteLLM(model=model)
        self.client = instructor.from_litellm(completion)

    async def astream(self, prompt: str) -> CompletionResponseAsyncGen:
        return await self.llm.astream_complete(prompt)

    def complete(self, prompt: str) -> CompletionResponse:
        return self.llm.complete(prompt)

    def structured_complete(self, response_model: type[T], prompt: str) -> T:
        print(prompt)
        return self.client.chat.completions.create(
            model=self.llm.model,
            response_model=response_model,
            messages=[{"content": prompt, "role": "user"}],
        )
