import warnings
warnings.filterwarnings("ignore")

import glob
from PIL import Image
from typing import Optional

from dotenv import load_dotenv
from transformers import pipeline

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# if running on collab
from MultiModalRAG.DataExtraction.extractionUtils import tables_text

# # otherwise 
# from extractionUtils import tables_text

load_dotenv()

def get_chain(api_key : Optional[str] = None) : 

    prompt_text = """
    You are an assistant tasked with summarizing tables and text.
    Give a concise summary of the table or text.

    Respond only with the summary, no additionnal comment.
    Do not start your message by saying "Here is a summary" or anything like that.
    Just give the summary as it is.

    Table or text chunk: {element}

    """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    if api_key : 

        model = ChatGroq(
            temperature=0.5,
            model="llama-3.1-8b-instant",
            groq_api_key = api_key 
        )
    
    else :  

        model = ChatGroq(
            temperature=0.5,
            model="llama-3.1-8b-instant"
        )
    
    chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    return chain

def get_summaries (tables, texts, api_key : Optional[str] = None) : 
    chain = get_chain(api_key)

    text_summary = chain.batch(texts, {"max_concurrency": 3})
    html_tables = [i.metadata.text_as_html for i in tables]
    table_summary = chain.batch(html_tables, {"max_concurrency": 3})

    return text_summary, table_summary

def encode_images(pipe):
    img_ls = []
    img_path_list = glob.glob("extracted/figure*jpg")

    for i in img_path_list:
        img = Image.open(i).convert('RGB')
        img_ls.append(pipe(img)[0]["generated_text"])

    return img_ls

def text_table_img (path, api_key : Optional[str] = None) : 
    tables, texts = tables_text(path)

    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device = 0)
    images = encode_images(pipe)


    text_summary, table_summary = get_summaries(tables, texts, api_key)

    return tables, texts, images , text_summary, table_summary, images
    
if __name__ == "__main__" : 
    path = 'data/document.pdf'

    model = ChatGroq(
        temperature=0.5,
        model="llama-3.1-8b-instant"    
    )
    print("done")
    chain = get_chain()
    
    tables, texts, images , text_summary, table_summary, image_summary= text_table_img(path)

