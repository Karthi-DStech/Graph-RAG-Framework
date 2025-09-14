# components/graph_ops.py
from typing import List, Optional, Any
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain


def create_knowledge_graph(
    *,
    docs: List[Document],
    opt,
    graph,
    llm: Any,
    include_source: bool = True,
) -> int:
    """
    Extract and ingest a knowledge graph from a list of LangChain Documents.

    Parameters
    ----------
    docs : List[Document]
        Preprocessed and chunked LangChain documents.
    opt : Any
        Parsed configuration object (argparse.Namespace or similar)
        containing optional settings like allowed_nodes,
        allowed_relationships, node_properties, and relationship_properties.
    graph : Any
        An active Neo4jGraph connection where graph documents will be ingested.
    llm : Any
        A pre-initialized LangChain-compatible chat model used for
        entity/relation extraction.
    include_source : bool, default=True
        Whether to store source metadata when adding graph documents.

    Returns
    -------
    int
        Number of graph documents ingested.

    Raises
    ------
    RuntimeError
        If the graph connection is missing or ingestion fails.
    """
    if graph is None:
        raise RuntimeError("Graph not connected")

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=getattr(opt, "allowed_nodes", None),
        allowed_relationships=getattr(opt, "allowed_relationships", None),
        node_properties=getattr(opt, "node_properties", False),
        relationship_properties=getattr(opt, "relationship_properties", False),
    )

    graph_documents = transformer.convert_to_graph_documents(docs)
    graph.add_graph_documents(graph_documents, include_source=include_source)
    return len(graph_documents)


def cypher_qa(
    *,
    question: str,
    opt,
    graph,
    llm: Any,
    verbose: bool = True,
) -> str:
    """
    Answer a natural-language question by generating and executing a Cypher query.

    The function:
        1. Retrieves the current Neo4j graph schema.
        2. Prompts the LLM to generate a valid Cypher query.
        3. Executes the query and returns the textual result.

    Parameters
    ----------
    question : str
        Natural-language question to translate into a Cypher query.
    opt : Any
        Parsed configuration object (argparse.Namespace or similar),
        included here for future extensibility.
    graph : Any
        An active Neo4jGraph connection to execute the Cypher query against.
    llm : Any
        A pre-initialized LangChain-compatible chat model used to generate Cypher.
    verbose : bool, default=True
        Whether to log the generated query and chain output.

    Returns
    -------
    str
        The textual answer returned from the Cypher query.

    Raises
    ------
    RuntimeError
        If the graph connection is missing or query execution fails.
    """
    if graph is None:
        raise RuntimeError("Graph not connected")

    schema = graph.get_schema()

    template = """
Task: Generate a Cypher statement to query the graph database.

Instructions:
- Use only relationship types and properties provided in the schema.
- Do not invent labels/relations/properties.

schema:
{schema}

# Output Rules:
- Output ONLY the Cypher query (no explanations).
- If the question cannot be answered with the schema, output a valid Cypher that returns an empty result with a clear RETURN.

Question: {question}
""".strip()

    cypher_prompt = PromptTemplate(
        template=template, input_variables=["schema", "question"]
    )

    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=cypher_prompt,
        verbose=verbose,
        allow_dangerous_requests=True,
    )

    res = chain.invoke({"schema": schema, "question": question})
    return res.get("result", "")
