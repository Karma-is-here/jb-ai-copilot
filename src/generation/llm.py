import os

from huggingface_hub import InferenceClient


class LLMClient:
    """
    Provider-agnostic LLM client.

    Supported providers:
        - hf: Hugging Face Inference Providers
    """

    def __init__(self):
        self.provider = os.getenv(
            "LLM_PROVIDER",
            "hf",
        )

        self.model = os.getenv(
            "LLM_MODEL",
            "openai/gpt-oss-120b",
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        if self.provider == "hf":
            return self._generate_huggingface(
                system_prompt,
                user_prompt,
            )

        raise ValueError(
            f"Unsupported LLM provider: {self.provider}"
        )

    def _generate_huggingface(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:

        token = os.getenv("HF_TOKEN")

        if not token:
            raise RuntimeError(
                "HF_TOKEN environment variable is not set."
            )

        client = InferenceClient(
            api_key=token,
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        return response.choices[0].message.content