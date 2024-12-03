import warnings
warnings.filterwarnings("ignore")
import tqdm
import glob
from PIL import Image
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

load_dotenv()

def partition (file_path) : 

    chunks = partition_pdf(
        filename=file_path,
        infer_table_structure=True,
        strategy="hi_res",
    
        extract_image_block_types=["Image"],
        extract_image_block_output_dir="extracted",
        extract_image_block_to_payload=False,

        chunking_strategy="by_title",           # ensures proper chunking based on titles
        max_characters=10000,
        combine_text_under_n_chars=2000,
        new_after_n_chars=6000,
    )
    return chunks

def tables_text (path) : 

    print("Chunking started")
    chunks = tqdm.tqdm(partition (file_path = path))
    print("Chunking Ended")
    
    tables = []
    texts = []

    for chunk in chunks:
        if "Table" in str(type(chunk)):
            tables.append(chunk)

        if "CompositeElement" in str(type((chunk))):
            texts.append(chunk)

    print("Text and Tables extracted")
    return tables, texts 

if __name__ == "__main__" : 
    path = 'data/document.pdf'
    tables, texts = tables_text(path)