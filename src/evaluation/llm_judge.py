from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def llm_judge_pipeline(model_name='gpt-4o-mini', temperature=0):
    """Cria o pipeline de avaliação LLM as a Judge."""
    prompt_template = """
    Given the following QUESTION, DOCUMENT and ANSWER you must analyze the provided answer and determine whether it is faithful to the contents of the DOCUMENT. The ANSWER must not offer new information beyond the context provided in the DOCUMENT. The ANSWER also must not contradict information provided in the DOCUMENT. Output your final verdict by strictly following this format: "PASS" if the answer is faithful to the DOCUMENT and "FAIL" if the answer is not faithful to the DOCUMENT. Show your reasoning.

    --
    QUESTION (THIS DOES NOT COUNT AS BACKGROUND INFORMATION):
    {question}

    --
    DOCUMENT:
    {context}

    --
    ANSWER:
    {answer}

    --

    Your output should be in JSON FORMAT with the keys "REASONING" and "SCORE":
    {{"REASONING": <your reasoning as bullet points>, "SCORE": <your final score>}}
    """
    llm = ChatOpenAI(model=model_name, temperature=temperature)
    hallucination_check_prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "context", "answer"]
    )
    return hallucination_check_prompt | llm | JsonOutputParser()

def evaluate_llm_judge(dataset, llm_chain):
    """Avalia o dataset usando LLM as a Judge."""
    scores = []
    reasonings = []
    for i in range(len(dataset)):
        result = llm_chain.invoke({
            "question": dataset['question'][i],
            "context": dataset['context'][i],
            "answer": dataset['llm_answer'][i],
        })
        scores.append(result.get('SCORE'))
        reasonings.append(result['REASONING'])
    dataset['score_llm_judge'] = scores
    dataset['reasoning_llm_judge'] = reasonings
    return dataset
