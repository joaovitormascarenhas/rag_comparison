import pandas as pd
from src.baseline import run_baseline
from src.parent_document import run_parent_document

def collect_responses(graph, questions):
    """Executa as perguntas em um grafo e coleta respostas."""
    responses = []
    for question in questions:
        response = graph.invoke({'question': question})
        responses.append({
            "question": question,
            "context": [doc.page_content for doc in response["context"]],
            "answer": response["answer"]
        })
    return responses

def save_results_to_dataframe(responses, method_name):
    """Transforma as respostas coletadas em um DataFrame."""
    data = []
    for response in responses:
        data.append({
            "method": method_name,
            "question": response["question"],
            "context": " ".join(response["context"]),
            "answer": response["answer"]
        })
    return pd.DataFrame(data)

def main():
    # Carregar dataset de perguntas
    qa_dataset = pd.read_csv('data/dataset_natal.csv')
    questions = qa_dataset['question'].tolist()

    # Inicializar os métodos
    baseline_graph = run_baseline('data/documents_natal.pkl')
    parent_document_graph = run_parent_document('data/documents_natal.pkl')

    # Coletar respostas do Baseline
    print("Executando perguntas no método Baseline...")
    baseline_responses = collect_responses(baseline_graph, questions)
    baseline_results = save_results_to_dataframe(baseline_responses, "Baseline")

    # Coletar respostas do Parent Document
    print("Executando perguntas no método Parent Document...")
    parent_responses = collect_responses(parent_document_graph, questions)
    parent_results = save_results_to_dataframe(parent_responses, "Parent Document")

    # Concatenar os resultados
    final_results = pd.concat([baseline_results, parent_results], ignore_index=True)

    # Salvar resultados em um arquivo
    final_results.to_pickle('data/rag_comparison_results.pkl', index=False)
    print("Resultados salvos em 'data/rag_comparison_results.pkl'")

if __name__ == "__main__":
    main()
