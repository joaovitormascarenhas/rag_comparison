from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import Faithfulness
from langchain_openai import ChatOpenAI

def ragas_faithfulness_pipeline(model_name='gpt-4o-mini'):
    """Cria o pipeline de avaliação RAGAS faithfulness."""
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=model_name))
    return Faithfulness(llm=evaluator_llm)

def evaluate_ragas_faithfulness(dataset, faithfulness_metric):
    """Avalia o dataset usando RAGAS faithfulness."""
    scores = []
    for i in range(len(dataset)):
        sample = SingleTurnSample(
            user_input=dataset['question'][i],
            response=dataset['llm_answer'][i],
            retrieved_contexts=[doc.page_content for doc in dataset['context'][i]]
        )
        score = await faithfulness_metric.single_turn_ascore(sample=sample)
        scores.append(score)
    dataset['score_ragas_llm_judge'] = scores
    dataset['tag_ragas_llm_judge'] = 'FAIL'
    dataset.loc[dataset['score_ragas_llm_judge'] >= 0.5, 'tag_ragas_llm_judge'] = 'PASS'
    return dataset
