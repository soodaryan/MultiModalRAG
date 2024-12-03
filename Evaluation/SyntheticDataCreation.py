import warnings
warnings.filterwarnings("ignore")

import json
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def initialize_settings():

    llm = Groq(model="llama3-8b-8192")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)

    return llm, embed_model, node_parser


llm , embed_model, node_parser = initialize_settings()


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)


file_path = "data/document.pdf"
docs_processed = []
documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

langchain_docs = [
    doc.to_langchain_format() 
    for doc in tqdm(documents)
]

for doc in langchain_docs:
    docs_processed += text_splitter.split_documents([doc])
print("done")


QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""


def chat(llm, query) : 
    """LLM caller. Returns the ouput to Query by calling the RAG agent."""
    resp = llm.complete(query)
    return str(resp)


from huggingface_hub import InferenceClient


repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
repo_id = "meta-llama/Llama-3.1-8B"
llm = InferenceClient(
    model=repo_id,
    timeout=120,
)


def chat(inference_client: InferenceClient, prompt: str):
    response = inference_client.post(
        json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 1000},
            "task": "text-generation",
        },
    )
    return json.loads(response.decode())[0]["generated_text"]


print(chat(llm, "This is a test context"))


import random

N_GENERATIONS = 10  
print(f"Generating {N_GENERATIONS} QA couples...")

outputs = []
for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS)):
    # Generate QA couple
    output_QA_couple = chat(
        llm, QA_generation_prompt.format(context=sampled_context.page_content)
    )
    try:
        question = output_QA_couple.split("Factoid question: ")[-1].split("Answer: ")[0]
        answer = output_QA_couple.split("Answer: ")[-1]
        assert len(answer) < 300, "Answer is too long"
        outputs.append(
            {
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
            }
        )
    except Exception as e:
        continue
question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to Eligible bidders for NIT Jalandhar e-tenders..
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independant this question is.
Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

The questions may include technical terms or acronyms like "EMD," "DPIIT," "Annexure-G," or "CPP Portal" and still qualify as a 5, as long as they are clear to someone familiar with procurement processes or tender documentation.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

print("Generating critique for each QA couple...")

for output in tqdm(outputs):
    evaluations = {
        "groundedness": chat(
            llm,
            question_groundedness_critique_prompt.format(
                context=output["context"], question=output["question"]
            ),
        ),
        "relevance": chat(
            llm,
            question_relevance_critique_prompt.format(question=output["question"]),
        ),
        "standalone": chat(
            llm,
            question_standalone_critique_prompt.format(question=output["question"]),
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            ) 
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except :
        try : 
            for criterion, evaluation in evaluations.items():
                score, eval = (
                    int(evaluation.split("Total rating: ")[-1].strip()[0]),
                    evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                ) 
                output.update(
                    {
                        f"{criterion}_score": score,
                        f"{criterion}_eval": eval,
                    }
                )
        except : 
            continue

import json

with open("Evaluation/questions.json", "w") as file:
    json.dump({"outputs" : outputs}, file, indent = 4) 


def filter_max_length_items(input_list):
    max_length = max(len(item) for item in input_list)  # Find the maximum length
    return [item for item in input_list if len(item) == max_length]  # Filter items with max length

outputs = filter_max_length_items(outputs)

import pandas as pd

generated_questions = pd.DataFrame.from_dict(outputs)

generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 4)
    & (generated_questions["relevance_score"] >= 4)
    & (generated_questions["standalone_score"] >= 4)
]

# generated_questions.to_csv("Evaluation/filtered_question.csv")
with open("Evaluation/filtered_questions.json", "w") as file:
    json.dump({"filtered_outputs" : generated_questions.to_json()}, file, indent = 4) 