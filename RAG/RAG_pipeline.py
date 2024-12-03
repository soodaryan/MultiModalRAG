import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
from RAG.ragUtils import makeVectoreStore
from DataExtraction.dataPreprocessing import text_table_img
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Optional


def load_api_key():
    load_dotenv()
    return os.getenv("GROQ_API_KEY")


def initialize_retriever(path, api_key : Optional[str] = None) :
    """
    """

    if api_key : 

        tables, texts, images, text_summary, table_summary, images_summary = text_table_img(path, api_key=api_key)
    
    else :  

        tables, texts, images, text_summary, table_summary, images_summary = text_table_img(path)

    
    return makeVectoreStore(tables, texts, images, table_summary, text_summary, images_summary)


def initialize_llm(temperature = 0.0):
    return ChatGroq(
        model="mixtral-8x7b-32768",
        temperature= temperature,
        max_retries=2,
        groq_api_key="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"
    )


def parse_docs(docs):
    return {"texts": docs}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]
    context_text = "".join(text_element.text for text_element in docs_by_type["texts"])
    prompt_template = f"Answer the question based only on the following context, which can include text and tables given below.\nContext: {context_text}\nQuestion: {user_question}"
    return ChatPromptTemplate.from_messages([HumanMessage(content=[{"type": "text", "text": prompt_template}])])


def create_chain(retriever, llm):
    chain = (
        {
            "context": retriever | RunnableLambda(parse_docs),
            "question": RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )
    return chain

def create_rephrase_history_chain(llm_pipeline, retriever, system_prompt):
    """
    Creates a history aware retriever. It is needed to generate comprehensive summary of the chat history while also considering the user 
    queries. 
    """
    contextualize_query_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                               MessagesPlaceholder("chat_history"),
                                                               ("human", "{input}")])

    history_aware_retriever = create_history_aware_retriever(llm_pipeline, retriever, contextualize_query_prompt)

    return history_aware_retriever

def create_qa_RAG_chain_history(llm_pipeline, retriever, system_prompt):
    """
    Performs RAG storing the chat history for future queries needed in conversational RAG.
    """
    qa_prompt = ChatPromptTemplate.from_messages([("system", system_prompt),
                                                  MessagesPlaceholder("chat_history"),
                                                  ("human", "{input}")])

    question_answer_chain = create_stuff_documents_chain(llm_pipeline, qa_prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain

def main():
    api_key = load_api_key()
    path = "data/document.pdf"
    retriever = initialize_retriever(path, api_key)
    llm = initialize_llm()
    chain = create_chain(retriever, llm)
    queries = [
        "What is the e-Tender notice number and the purpose of the tender mentioned in the document?",
        "What are the eligibility criteria for bidders to participate in this tender?",
        "What are the deadlines for submitting the online bids and physically submitting the tender fee and EMD?",
        "What is the role of Annexure-G in determining supplier eligibility, and how is local content defined?",
        "What is the payment structure for the successful supplier as mentioned in the document?",
        "What are the warranty obligations for suppliers as outlined in Annexure-F?",
        "How is the technical bid evaluated, and what criteria are used for shortlisting bidders?",
        "What penalties are imposed for delays in delivery or non-performance by the supplier?",
        "What does Annexure-E specify about the blacklisting or debarment of suppliers?",
        "What does Annexure-D require from suppliers regarding manufacturer authorization?"
    ]
    responses = {}

    for query in queries : 
        response = chain.invoke(query)
        responses[query] = response
        print(f"Q) {query}")
        print(f"A) {response}")
    

    for i in responses : 
        print(f"Q) {i}")
        print(f"A) {responses[i]}")
    
if __name__ == "__main__":
    main()
