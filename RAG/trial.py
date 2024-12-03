"""
Please note : this is a dummy document which can be used for rough code examples and testing 
"""



import warnings
warnings.filterwarnings("ignore")

import os 
from dotenv import load_dotenv
from RAG.ragUtils import makeVectoreStore
from DataExtraction.dataPreprocessing import text_table_img

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

path = 'data/document.pdf'
tables, texts, images , text_summary, table_summary, images_summary = text_table_img(path, api_key = GROQ_API_KEY)

retriever = makeVectoreStore (tables, texts, images , table_summary, text_summary, images_summary)
# docs = retriever.invoke( "what is the document about?")

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    temperature=0.0,
    max_retries=2,
    groq_api_key = "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"

)


from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# from langchain_openai import ChatOpenAI
# from base64 import b64decode


def parse_docs(docs):
    return {"texts": docs}


def build_prompt(kwargs):

    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images)
    prompt_template = f"""
    Answer the question based only on the following context, which can include text and tables given below.
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )


chain = (
    {
        "context": retriever | RunnableLambda(parse_docs),
        "question": RunnablePassthrough(),
    }
    | RunnableLambda(build_prompt)
    | llm
    | StrOutputParser()
)

chain_with_sources = {
    "context": retriever | RunnableLambda(parse_docs),
    "question": RunnablePassthrough(),
} | RunnablePassthrough().assign(
    response=(
        RunnableLambda(build_prompt)
        | llm
        | StrOutputParser()
    )
)

response = chain.invoke(
    "What is the tender fee amount, and is it refundable?"
)

print(response)

# tables, texts, images , text_summary, table_summary, images_summary = text_table_img(
#     path, 
#     api_key = groq_api_key
# )



