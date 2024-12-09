import warnings
warnings.filterwarnings("ignore")

import datasets
import random
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain.text_splitter import RecursiveCharacterTextSplitter
from OnDemandWrapper import chat
from prompts import get_qa_gen_prompt, get_critique_prompts
from collections import defaultdict


load_dotenv()

def initialize_models():
    llm = Groq(model="llama3-8b-8192")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    return llm, embed_model, node_parser

def setup_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        add_start_index=True,
        separators=["\n\n", "\n", ".", " ", ""],
    )

def load_documents(file_path):
    return SimpleDirectoryReader(input_files=[file_path]).load_data()

def split_documents(documents, text_splitter):
    split_docs = []
    for doc in tqdm(documents):
        split_docs += text_splitter.split_documents([doc.to_langchain_format()])
    return split_docs

def generate_qa_pairs(llm, docs, qa_prompt, n_pairs):
    extended_docs = docs * ((n_pairs // len(docs)) + 1)
    qa_outputs = []
    for sampled_context in tqdm(random.sample(extended_docs, n_pairs)):
        try:
            output = chat(llm, qa_prompt.format(context=sampled_context.page_content))
            question = output.split("Factoid question: ")[-1].split("Answer: ")[0]
            answer = output.split("Answer: ")[-1]
            assert len(answer) < 300, "Answer is too long"
            qa_outputs.append({
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
            })
        except Exception:
            continue
    return qa_outputs

def critique_qa_pairs(llm, outputs, critique_prompts):
    for output in tqdm(outputs):
        evaluations = {
            criterion: chat(
                llm, 
                prompt.format_map(
                    defaultdict(
                        lambda: "", {
                            "context": output.get("context", ""),
                            "question": output.get("question", "")
                            }
                        )
                    )
                ) for criterion, prompt in critique_prompts.items()
        }
        try:
            for criterion, evaluation in evaluations.items():
                score = int(evaluation.split("Total rating: ")[-1].strip())
                eval_text = evaluation.split("Evaluation: ")[1].split("Total rating:")[0]
                output[f"{criterion}_score"] = score
                output[f"{criterion}_eval"] = eval_text
        except Exception:
            continue
    return outputs

def filter_and_save_qa_pairs(outputs, save_path, score_threshold=4):
    df = pd.DataFrame(outputs)
    filtered_df = df.loc[
        (df["groundedness_score"] >= score_threshold) &
        (df["relevance_score"] >= score_threshold) &
        (df["standalone_score"] >= score_threshold-1)
    ]
    filtered_df.to_json(save_path, orient="records", indent=4)
    return filtered_df


def main():
    llm, embed_model, node_parser = initialize_models()
    text_splitter = setup_text_splitter()
    
    file_path = "data/document.pdf"
    documents = load_documents(file_path)
    docs_processed = split_documents(documents, text_splitter)
    
    qa_prompt = get_qa_gen_prompt()
    
    n_pairs = 100
    print(f"Generating {n_pairs} QA pairs...")
    qa_outputs = generate_qa_pairs(llm, docs_processed, qa_prompt, n_pairs)
    
    critique_prompts = get_critique_prompts()
    
    print("Critiquing QA pairs...")
    qa_outputs = critique_qa_pairs(llm, qa_outputs, critique_prompts)
    
    save_path = "Evaluation/filtered_questions.json"
    print(f"Saving filtered QA pairs to {save_path}...")
    filtered_qa_pairs = filter_and_save_qa_pairs(qa_outputs, save_path)

    print(f"Number of filtered questions: {len(filtered_qa_pairs)}")

    eval_dataset = datasets.Dataset.from_pandas(
        filtered_qa_pairs, split="train", preserve_index=False
    )
    eval_dataset.save_to_disk("Evaluation/")


if __name__ == "__main__":
    main()
