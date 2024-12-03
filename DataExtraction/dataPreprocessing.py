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
from langchain_core.output_parsers import StrOutputParser

# # if running on collab
# from MultiModalRAG.DataExtraction.extractionUtils import tables_text

# otherwise 
from DataExtraction.extractionUtils import tables_text

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
    imgs = []
    img_path_list = glob.glob("extracted/figure*jpg")

    reader = easyocr.Reader(['en']) 

    for i in img_path_list:
        img = Image.open(i).convert('RGB')
        img_ls.append(pipe(img)[0]["generated_text"])

        results = reader.readtext(i)
        extracted_text = ""

        # Print the results
        for _ , text , _ in results:
            extracted_text += f" {text}"
        imgs.append(extracted_text)

    return imgs, img_ls

def text_table_img (path, api_key : Optional[str] = None) : 
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
