import warnings
warnings.filterwarnings("ignore")

import glob
import easyocr
from PIL import Image
from typing import Optional

from dotenv import load_dotenv
from transformers import pipeline

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from DataExtraction.extractionUtils import tables_text
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_chain(api_key : Optional[str] = None) : 
    """
    Creates a chain for RAG agent using suitable prompt.
    :param api_key: optional api key for Groq
    :return: Runable Chain
    """
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
    """
    Generates table and text summaries used for efficient data retrieval.
    :param tables: list of tables in document
    :param texts: list of text paragraphs in document
    :return: summaries of text and tables
    """
    chain = get_chain(api_key)

    text_summary = chain.batch(texts, {"max_concurrency": 3})
    html_tables = [i.metadata.text_as_html for i in tables]
    table_summary = chain.batch(html_tables, {"max_concurrency": 3})

    return text_summary, table_summary

def encode_images(pipe):
    """
    Generates image captions summaries used for all images extracted from the document ensuring efficient data retrieval.
    :param pipe: pipeline for BLIP/CLIP model 
    :return: image captions list
    """
    img_ls = []
    imgs = []
    img_path_list = glob.glob("extracted/figure*jpg")

    reader = easyocr.Reader(['en']) 

    for i in img_path_list:
        img = Image.open(i).convert('RGB')
        img_ls.append(pipe(img)[0]["generated_text"])

        results = reader.readtext(i)
        extracted_text = ""

        for _ , text , _ in results:
            extracted_text += f" {text}"
        imgs.append(extracted_text)

    return imgs, img_ls

def text_table_img (path, api_key : Optional[str] = None) : 
    """
    Wrapper function for full data extraction and preprocessing
    :param path: path of file/doc
    :api_key: optional input for Groq API key
    :return: extracted information namely tables, texts, images, text_summary, table_summary, images_summary
    """
    tables, texts = tables_text(path)

    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large", device = 0)
    images, images_summary = encode_images(pipe)
    print("Images extracted")

    text_summary, table_summary = get_summaries(tables, texts, api_key)
    print("Text, Table and Image Summaries extracted")

    return tables, texts, images, text_summary, table_summary, images_summary
    
if __name__ == "__main__" : 

    path = 'data/document.pdf'
    tables, texts, images , text_summary, table_summary, image_summary= text_table_img(path)
    print(f"tables : {len(tables)}")
    print(f"tables : {len(table_summary)}")
