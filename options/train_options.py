from options.base_options import BaseOptions
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TrainOptions(BaseOptions):
    """
    This class defines the train options for the script.

    Parameters
    ----------
    None
    """

    def __init__(self) -> None:
        super().__init__()

    def initialize(self) -> None:
        """
        Initialize train options
        """
        BaseOptions.initialize(self)

        # -------- OpenAI / LangChain --------

        self.parser.add_argument(
            "--LLMProvider",
            type=list,
            default=["OpenAI", "Antropic", "Google"],
            help="LLM provider to use: OpenAI, Antropic, Google",
        )

        self.parser.add_argument(
            "--OpenAILLM",
            type=list,
            default=["gpt-4o", "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
            help="OpenAI chat model for Cypher query generation",
        )

        self.parser.add_argument(
            "--AntropicILLM",
            type=list,
            default=["claude-2", "claude-instant-100k", "claude-1", "claude-1.3"],
            help="Antropic chat model for Cypher query generation",
        ),

        self.parser.add_argument(
            "--GoogleLLM",
            type=list,
            default=["claude-2", "claude-instant-100k", "claude-1", "claude-1.3"],
            help="Google chat model for Cypher query generation",
        ),

        self.parser.add_argument(
            "--EmbeddingModel",
            type=str,
            default="OpenAI",
            help="Embedding model name (defaults to OpenAIEmbeddings default)",
        )

        # -------- PDF Text Splitting --------

        self.parser.add_argument(
            "--ChunkSize",
            type=int,
            default=200,
            help="Character chunk size for text splitting",
        )
        self.parser.add_argument(
            "--ChunkOverlap",
            type=int,
            default=40,
            help="Overlap size between consecutive chunks",
        )

        # -------- Knowledge Graph Extraction --------

        self.parser.add_argument(
            "--AllowedNodes",
            type=str,
            default="Patient,Disease,Medication,Test,Symptom,Doctor",
            help="Comma-separated list of allowed node labels",
        )
        self.parser.add_argument(
            "--AllowedRelationships",
            type=str,
            default=[],
            help="Comma-separated list of allowed relationship types",
        )
        self.parser.add_argument(
            "--NodeProperties",
            type=bool,
            default=False,
            help="Include node properties in the graph (True/False)",
        )
        self.parser.add_argument(
            "--RelationshipProperties",
            type=bool,
            default=False,
            help="Include relationship properties in the graph (True/False)",
        )
        self.parser.add_argument(
            "--ClearGraph",
            type=bool,
            default=True,
            help="Clear existing Neo4j graph before ingestion (True/False)",
        )

        # -------- Neo4j Vector / Full-Text Indexing --------

        self.parser.add_argument(
            "--NodeLabel",
            type=str,
            default="Patient",
            help="Node label to use when creating the vector index",
        )
        self.parser.add_argument(
            "--TextNodeProperties",
            type=str,
            default="id,text",
            help="Comma-separated list of text properties for keyword index",
        )
        self.parser.add_argument(
            "--EmbeddingNodeProperty",
            type=str,
            default="embedding",
            help="Node property name to store embeddings",
        )
        self.parser.add_argument(
            "--VectorIndexName",
            type=str,
            default="vector_index",
            help="Name of the Neo4j vector index",
        )
        self.parser.add_argument(
            "--KeywordIndexName",
            type=str,
            default="entity_index",
            help="Name of the Neo4j keyword/full-text index",
        )
        self.parser.add_argument(
            "--SearchType",
            type=str,
            default="hybrid",
            choices=["vector", "keyword", "hybrid"],
            help="Retrieval search type: vector, keyword, or hybrid",
        )
