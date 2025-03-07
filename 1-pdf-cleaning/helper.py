# loading packages
import pymupdf, pymupdf4llm
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

def extract_metadata(file_path):
    """extracts metadata from PDF file

    Args:
        file_path (str): string of file path of PDf

    Returns: 
        metadata (dict): a dictionary containing the following info about the
                        original PDF: title, author, date of creation, subject, 
                                        keywords, format
    """

    # list of metadata to extract
    metadata_to_extract = ["title", "author", "creationDate", 
                           "subject", "keywords", "format"]

    # all metadata extracted by pymupdf
    all_metadata = pymupdf.open(file_path).metadata
    
    # subsetting desired metadata
    metadata = {data: all_metadata[data] for data in metadata_to_extract}

    return metadata

def clean_text(text):
    """cleans text for LLM/RAG models

    Args:
        text (str): string containing text to be cleaned
    
    Returns:
        text (str): string containing cleaned text
    """
    text = re.sub(r'[^\w\s]', '', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace

    # Remove common research paper patterns
    text = re.sub(r'\[\d+\]', '', text) # Remove citation brackets like [1]
    text = re.sub(r'\(.*?et al., \d{4}\)', '', text) # Remove author citations

    return text

def extract_text(file_path):
    """extracts raw text from PDF, cleans it, and splits into chunks

    Args:
        file_path (str): string of file path
    
    Returns:
        text (str): string containing all text in PDF
    """

    # converts PDf to markdown; then extracts text
    text = pymupdf4llm.to_markdown(file_path)
    
    # clean text
    cleaned_text = clean_text(text)

    # defines splitter for chunking texts
    splitter = RecursiveCharacterTextSplitter(chunk_size = 512, 
                                               # maintain context between chunks
                                              chunk_overlap = 50, 
                                              # avoid splitting in the middle of paragraph/sentence/word;
                                              # splits text at end of paragraph/sentence/word
                                              separators=["\n\n", "\n", " ", ""] 
                                              )
    
    return splitter.split_text(cleaned_text)

# Prepare Data ------------------------------------------------------------------

# uncomment code below to run helper.py to clean and extract data from PDFs 
# (without using pdf_cleaning.ipynb)

# # file path for all research paper PDFs
# file_paths = ["raw-data/1. A review of deep learning-based stereo vision techniques.pdf",
#               "raw-data/2. Application of Biofloc technology in shrimp aquaculture A review on.pdf",
#               "raw-data/3. Could Biofloc Technology (BFT) Pave the Way Toward a More.pdf",
#               "raw-data/4. Fish gut microbiome and its application in aquaculture and biological conservation.pdf",
#               "raw-data/5. Genome Manipulation Advances in Selected.pdf"]

# # pymupdf couldn't extract title from this PDF
# stereo_vision_title = "A review of deep learning-based stereo vision techniques for phenotype feature and behavioral analysis of fish in aquaculture"

# # initialize list to store metadata dict for each PDF
# cleaned_text = []
# count = 0

# for path in file_paths: # extract metadata from each PDF
#     cleaned_text.append(extract_metadata(path))
#     cleaned_text[count]["text"] = extract_text(path)
#     count += 1

# # manually adding title of first PDF 
# cleaned_text[0].update({"title": stereo_vision_title})

# # save results as jsonl file
# with open('cleaned_research_paper.jsonl', 'w') as file:
#     # write each dictionary as a separate line in final JSONL file
#     for paper in cleaned_text:
#         # convert dict into JSON string
#         json_string = json.dumps(paper)
#         # write to file with newline separator
#         file.write(json_string + '\n')