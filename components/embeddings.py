from typing import Any, Optional


class GetEmbeddings:
    """
    Factory class to create a LangChain embedding model instance
    based on the configuration options or an explicitly provided name.

    Parameters
    ----------
    opt : argparse.Namespace or similar
        Parsed options object that must contain an attribute
        `EmbeddingModel` (string) if `embedding_name` is not supplied
        to `get_embedding_models`.
    """

    def __init__(self, opt) -> None:
        """
        Store the options object for later use.

        Parameters
        ----------
        opt : argparse.Namespace
            Parsed command-line or configuration options.
        """
        self._opt = opt

    def get_embedding_models(self, embedding_name: Optional[str] = None) -> Any:
        """
        Return an initialized LangChain embeddings model.

        If `embedding_name` is provided it takes precedence; otherwise
        the value of `self._opt.EmbeddingModel` is used.

        Parameters
        ----------
        embedding_name : str, optional
            Explicit embedding provider name. If omitted, falls back to
            `self._opt.EmbeddingModel`.

        Returns
        -------
        Any
            An instantiated LangChain embeddings object ready to use.

        Raises
        ------
        ValueError
            If no provider is specified or the name is not recognized.
        RuntimeError
            If instantiation of the chosen embedding model fails.
        """

        embedding_name = embedding_name or getattr(self._opt, "EmbeddingModel", None)
        if not embedding_name:
            raise ValueError(
                "No embedding provider specified (embedding_name or opt.EmbeddingModel)."
            )
        embedding_model = None

        try:

            if embedding_name == "openai":
                from langchain.embeddings import OpenAIEmbeddings

                embedding_model = OpenAIEmbeddings()

            elif embedding_name == "cohere":
                from langchain.embeddings import CohereEmbeddings

                embedding_model = CohereEmbeddings()

            elif embedding_name == "google_palm":
                from langchain.embeddings import GooglePalmEmbeddings

                embedding_model = GooglePalmEmbeddings()

            elif embedding_name == "vertexai":
                from langchain.embeddings import VertexAIEmbeddings

                embedding_model = VertexAIEmbeddings()

            elif embedding_name == "bedrock":
                from langchain.embeddings import BedrockEmbeddings

                embedding_model = BedrockEmbeddings()

            else:
                raise ValueError(
                    f"Unknown embedding provider: {self._opt.EmbeddingModel}"
                )

            return embedding_model

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize embeddings for {embedding_name}: {e}"
            )
