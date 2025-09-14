# components/neo4j_store.py
import argparse
import os
import sys
from typing import List, Optional, Optional, Tuple
from langchain.graphs import Neo4jGraph
from langchain.vectorstores import Neo4jVector
from langchain.schema import Document

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Neo4jStore:
    """
    Wrapper for managing a Neo4j database connection and vector/keyword indexes.

    Parameters
    ----------
    opt : argparse.Namespace
        Parsed CLI/config options containing Neo4j credentials and
        index settings (uri, username, password, database, etc.).
    """

    def __init__(self, opt: argparse.Namespace) -> None:
        """
        Initialize the Neo4jStore with configuration options.

        Parameters
        ----------
        opt : argparse.Namespace
            Parsed configuration including Neo4j credentials.
        """
        self._opt = opt
        self.graph: Optional[Neo4jGraph] = None

    def _connection(self):
        """
        Establish a connection to the Neo4j database.

        Returns
        -------
        Neo4jGraph
            A connected Neo4jGraph instance.

        Raises
        ------
        RuntimeError
            If the connection attempt fails.
        """
        try:
            self.graph = Neo4jGraph(
                url=self._opt.uri,
                username=self._opt.username,
                password=self._opt.password,
                database=self._opt.database,
            )

            return self.graph

        except Exception as e:
            raise RuntimeError(f"Failed to connect to Neo4j: {e}")

    def clear(self):
        """
        Delete all nodes and relationships from the graph.

        Raises
        ------
        RuntimeError
            If the graph is not connected or the query fails.
        """
        try:
            if not self.graph:
                raise RuntimeError("Graph not connected")
            self.graph.query("MATCH (n) DETACH DELETE n;")

        except Exception as e:
            raise RuntimeError(f"Failed to clear Neo4j database: {e}")

    def add_graph_documents(
        self, graph_documents: List, include_source: bool = True
    ) -> None:
        """
        Add LangChain GraphDocuments to the Neo4j database.

        Parameters
        ----------
        graph_documents : List
            List of graph documents to ingest.
        include_source : bool, default=True
            Whether to include source metadata.

        Raises
        ------
        RuntimeError
            If the graph is not connected or ingestion fails.
        """
        try:
            if not self.graph:
                raise RuntimeError("Graph not connected")
            self.graph.add_graph_documents(
                graph_documents, include_source=include_source
            )

        except Exception as e:
            raise RuntimeError(f"Failed to add documents to Neo4j: {e}")

    def create_hybrid_indexes(self) -> Neo4jVector:
        """
        Create combined vector and keyword indexes for hybrid search.

        Returns
        -------
        Neo4jVector
            A Neo4jVector object bound to the created indexes.

        Raises
        ------
        RuntimeError
            If the graph is not connected or index creation fails.
        """

        if not self.graph:
            raise RuntimeError("Graph not connected.")

        try:

            self.vector = Neo4jVector.from_existing_graph(
                embedding=self._opt.embeddings,
                url=self._opt.uri,
                username=self._opt.username,
                password=self._opt.password,
                database=self._opt.database,
                node_label=self._opt.node_label,
                text_node_properties=self._opt.text_node_properties,
                embedding_node_property=self._opt.embedding_node_property,
                index_name=self._opt.index_name,
                keyword_index_name=self._opt.keyword_index_name,
                search_type=self._opt.search_type,
            )

            return self.vector

        except Exception as e:
            raise RuntimeError(f"Failed to create hybrid indexes: {e}")

    def similarity_search(
        self, query: str, k: int = 5, with_score: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        """
        Perform pure vector similarity search.

        Parameters
        ----------
        query : str
            Search query text.
        k : int, default=5
            Number of top results to return.
        with_score : bool, default=False
            Whether to include similarity scores.

        Returns
        -------
        List[Document] | List[Tuple[Document, float]]
            Matching documents (and optional scores).

        Raises
        ------
        RuntimeError
            If the vector index is missing or search fails.
        """
        if not self.vector:
            raise RuntimeError(
                "Vector index not initialized. Call create_hybrid_indexes()."
            )
        try:
            if with_score:
                return self.vector.similarity_search_with_score(query, k=k)
            return self.vector.similarity_search(query, k=k)
        except Exception as e:
            raise RuntimeError(f"Similarity search failed: {e}")

    def hybrid_search(
        self, query: str, k: int = 5, with_score: bool = False
    ) -> List[Document] | List[Tuple[Document, float]]:
        """
        Perform hybrid (vector + keyword) search.

        Parameters
        ----------
        query : str
            Search query text.
        k : int, default=5
            Number of top results to return.
        with_score : bool, default=False
            Whether to include combined relevance scores.

        Returns
        -------
        List[Document] | List[Tuple[Document, float]]
            Matching documents (and optional scores).

        Raises
        ------
        RuntimeError
            If the vector index is missing or search fails.
        """
        if not self.vector:
            raise RuntimeError(
                "Vector index not initialized. Call create_hybrid_indexes()."
            )
        try:
            if with_score:
                return self.vector.hybrid_search_with_score(query, k=k)
            return self.vector.hybrid_search(query, k=k)
        except Exception as e:
            raise RuntimeError(f"Hybrid search failed: {e}")
