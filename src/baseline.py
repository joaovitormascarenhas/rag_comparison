import os
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict, List

from src.utils import load_dataset
from src.vectorstore import create_vectorstore

# Configurações de ambiente
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'rag_baseline'

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State, vectorstore):
    """Etapa de recuperação de documentos relevantes."""
    retrieved_docs = vectorstore.similarity_search(state["question"], k=3)
    return {"context": retrieved_docs}

def generate(state: State, prompt, llm):
    """Etapa de geração de resposta."""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def run_baseline(file_path):
    # Carregar documentos
    documents = load_dataset(file_path)

    # Criar banco de vetores
    vectorstore = create_vectorstore(documents)

    # Configurar prompt e modelo
    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    
    # Criar e compilar grafo de estados
    graph_builder = StateGraph(State).add_sequence([
        ("retrieve", lambda state: retrieve(state, vectorstore)),
        ("generate", lambda state: generate(state, prompt, llm))
    ])

    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    return graph
