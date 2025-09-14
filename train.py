from options.train_options import TrainOptions
from utils.preprocessing import DocumentProcessor
from components.embeddings import GetEmbeddings
from components.graph_db import Neo4jStore
from components.llms import GetLLM
from components.knowledge_graph import create_knowledge_graph, cypher_qa


def run():
    """
    Runs the RAG model.

    parameters
    ----------
    None

    Process
    -------

    """

    # 1. Parse options.
    opt = TrainOptions().parse()

    # 2. Preprocess documents.
    docs = DocumentProcessor(opt).process_documents()

    # 3. Connect Graph DB.
    store = Neo4jStore(opt)
    graph = store.connect()

    # 4. Initialize the LLMs.
    llms = GetLLM(opt).get_chat_model()

    # 5. Create Knowledge Graph from chunks (ingest)
    create_knowledge_graph(
        docs=docs,
        opt=opt,
        graph=graph,
        llm=llms,
        include_source=True,
    )

    # 6. Initialize embeddings and create hybrid search indexes (Vectors)
    opt.embeddings = GetEmbeddings(opt).get_embedding_models()
    store.create_hybrid_indexes()

    # 7. Test Cypher QA (Prompt user for a question)
    if opt.test_knowledge_graph:

        graph_question = input("Please enter your question: ")
        kg_answer = cypher_qa(
            question=graph_question,
            opt=opt,
            graph=graph,
            llm=llms,
            verbose=True,
        )
        print("QA:", kg_answer)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    run(opt=opt)
