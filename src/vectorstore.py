from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

def create_vectorstore(documents, model_name='BAAI/bge-m3', device='cuda'):
    """Cria e retorna um banco de vetores Chroma."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=bge_embeddings)
    return vectorstore

def create_parent_retriever(documents, model_name='BAAI/bge-m3', device='cuda'):
    """Cria e retorna o Parent Document Retriever."""
    # Splitters para documentos
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # Configurar embeddings
    bge_embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Banco de vetores e armazenamento em memória
    vectorstore = Chroma(collection_name="split_parents", embedding_function=bge_embeddings)
    store = InMemoryStore()

    # Criar ParentDocumentRetriever
    parent_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )
    parent_retriever.add_documents(documents)

    return parent_retriever
