import argparse
import os
import sys
from typing import Any, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class GetLLM:
    """
    Factory class for creating LangChain chat-based Large Language Model (LLM)
    instances from different providers such as OpenAI, Anthropic, or Google.

    Parameters
    ----------
    opt : argparse.Namespace
        Parsed options object that should contain:
        - LLMProvider : str   (e.g. "openai", "anthropic", "google")
        - LLMModel    : str   (e.g. "gpt-4", "claude-2", "gemini-pro")
        - temperature : float (sampling temperature)
        - max_tokens  : int   (max tokens in response)
        - timeout     : int   (request timeout in seconds)
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Store configuration options for later LLM instantiation.

        Parameters
        ----------
        opt : argparse.Namespace
            Command-line or configuration options.
        """
        self._opt = opt

    def get_chat_model(
        self, provider_name: Optional[str] = None, model_name: Optional[str] = None
    ) -> Any:
        """
        Instantiate and return a LangChain ChatModel based on the specified provider
        and model name.

        Parameters
        ----------
        provider_name : str, optional
            LLM provider name ("openai", "anthropic", or "google").

        model_name : str, optional
            Specific model name (e.g., "gpt-4", "claude-2", "gemini-pro").

        Returns
        -------
        Any
            An initialized LangChain chat model instance.

        Raises
        ------
        ValueError
            If required provider/model information is missing or unsupported.
        RuntimeError
            If the underlying LangChain model class cannot be initialized.
        """

        llm_name = provider_name or self._opt.LLMProvider.lower()
        model_name = model_name or self._opt.LLMModel

        if not llm_name or not model_name:
            raise ValueError(
                "No LLM provider or modelname specified. Given llm name is {lm_name} and model_name is {model_name})."
            )

        if llm_name == "openai" and model_name in [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4",
            "gpt-3.5-turbo",
        ]:
            try:
                from langchain.chat_models import ChatOpenAI

                return ChatOpenAI(
                    model_name=llm_name,
                    temperature=self._opt.temperature,
                    max_tokens=self._opt.max_tokens,
                    request_timeout=self._opt.timeout,
                )

            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI LLM: {e}")

        elif llm_name == "antropic" and model_name in [
            "claude-2",
            "claude-instant-100k",
            "claude-1",
            "claude-1.3",
        ]:
            try:
                from langchain.chat_models import ChatAnthropic

                return ChatAnthropic(
                    model=llm_name,
                    temperature=self._opt.temperature,
                    max_tokens=self._opt.max_tokens,
                    request_timeout=self._opt.timeout,
                )

            except Exception as e:
                raise RuntimeError(f"Failed to initialize Anthropic LLM: {e}")

        elif (
            llm_name == "google"
            and model_name["gemini-pro", "gemini-1.5", "gemini-1.0"]
        ):
            try:
                from langchain.chat_models import ChatGoogleGemini

                return ChatGoogleGemini(
                    model=llm_name,
                    temperature=self._opt.temperature,
                    max_tokens=self._opt.max_tokens,
                    request_timeout=self._opt.timeout,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Google Gemini LLM: {e}")

        else:
            raise ValueError(
                f"Unknown LLM provider: {llm_name} and model name: {model_name}"
            )
