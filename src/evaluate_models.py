from config import LLM_MODEL_NAME, TEMPERATURE
import pandas as pd
from utils.dataset_utils import load_dataset, save_dataset
from evaluation.llm_judge import llm_judge_pipeline, evaluate_llm_judge
from evaluation.ragas_faithfulness import ragas_faithfulness_pipeline, evaluate_ragas_faithfulness

# Carregar dataset
dataset_path = "data/rag_comparison_results.pkl"
dataset = pd.read_pickle(dataset_path)

# Avaliação LLM as a Judge
llm_pipeline = llm_judge_pipeline('gpt-4o-mini', 0)
dataset = evaluate_llm_judge(dataset, llm_pipeline)

# Avaliação RAGAS Faithfulness
ragas_pipeline = ragas_faithfulness_pipeline('gpt-4o-mini')
dataset = evaluate_ragas_faithfulness(dataset, ragas_pipeline)

# Salvar resultados
output_path = "data/rag_evaluate_scores.pkl"
save_dataset(dataset, output_path)
