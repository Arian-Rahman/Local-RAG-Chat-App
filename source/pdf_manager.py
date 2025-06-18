import pdfplumber
import os
import pathlib
import google.generativeai as genai
from dotenv import load_dotenv
import logging
#import utils.paths as paths
import source.utils.paths as paths


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


extracted_text_path = paths.paths["extracted_data"]


def configure_google_gemini():
    load_dotenv() 
    api_key = os.getenv("GOOGLE_GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_GEMINI_API_KEY not found in environment.")
    genai.configure(api_key=api_key)
    
def inspect_pdf(pdf_path):
    """
    Inspects PDF and returns metadata about its content.
    """
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        has_text , has_images , has_tables = False, False, False
        for page_num,page in enumerate(pdf.pages):
            text = page.extract_text()
            tables = page.extract_tables()
            if text and text.strip():
                has_text = True
            if tables and any(tables): 
                has_tables = True
            
            if page.objects.get("image"):
                has_images = True
                
            if has_text and has_images and has_tables:
                break
        return {
            "pdf_path": pdf_path,
            "num_pages": num_pages,
            "has_text": has_text,
            "has_images": has_images,
            "has_tables": has_tables
        }
        
def is_complex_table(table):
    """
    Checks for complex tables that have inconsistent row lengths.
    """
    num_of_cols_for_first_row = len(table[0]) if table else 0
    for row in table:
        if len(row) != num_of_cols_for_first_row:
            return True

def format_table_as_text(table):
    """
    Formats a table into a string representation using pipes and row separators.
    """
    formatted_rows = [] 
    for row in table : 
        formatted_row = "|".join(cell.strip() if cell else "" for cell in row  )
        formatted_rows.append(formatted_row)
        formatted_rows.append("-" * len(formatted_row))  
    
    return "\n".join(formatted_rows)


def extract_using_LLM(pdf_path):
    """
    Extracts contents from a PDF
    using the Google Gemini API.
    """
    
    logger.info(f"Extracting data from {pdf_path} using Google Gemini API...")    
    
    filepath = pathlib.Path(pdf_path)
    pdf_data = filepath.read_bytes()
    
    configure_google_gemini()
    model = genai.GenerativeModel("models/gemini-1.5-flash")  

    prompt = (
        "You are an expert document parser.\n"
        "Please extract all readable text from this PDF, including:\n"
        "1. All plain text content\n"
        "2. Properly formatted tables using pipes (|) and row separators\n"
        "3. Any text from images using OCR\n , the data from image might not be table, so just extract the text in appropriate format .\n"
        "Return everything in a clean, readable format."
    )

    response = model.generate_content([
        {"role": "user", "parts": [prompt, {"mime_type": "application/pdf", "data": pdf_data}]}
    ])

    if response and getattr(response, "text", None):
        return {
            "file_name": filepath.name,
            "data": response.text.strip()
        }
    else:
        logger.error(f"Failed to extract data from {pdf_path} using Gemini. Error: {getattr(response, 'error', 'No response')}")
        return None
   
              
        
def extract_from_text_pdf(pdf_path):
    """
    Extracts text locally from a PDF file that contains only text.
    """
    file_name = os.path.basename(pdf_path)
    pdf_data = {"file_name": file_name, "data" : ""}
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
                
    pdf_data["data"] = "\n".join(all_text)
    print(f"Extracted text from {file_name}")
    logger.info(f"Successfully Extracted text from {file_name}")
    return pdf_data
        

def extract_from_table_pdf(pdf_path):
    """
    Extracts table contents locally from pdf
    """
    file_name = os.path.basename(pdf_path)
    pdf_data = {"file_name": file_name, "data": ""}
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages):
            page_content = [f"___Page {page_number + 1}___"]

            # Extracting tables first and using sets to avoid duplicates later
            tables = page.extract_tables()
            table_contents_set = set() 
            formatted_tables = []

            if tables:
                for i, table in enumerate(tables):
                    if is_complex_table(table):
                        llm_extracted_data = extract_using_LLM(pdf_path)
                        return llm_extracted_data
                        
                    formatted_table = format_table_as_text(table)
                    formatted_tables.append(f"\n[Page {page_number+1}] [Table {i+1}]\n{formatted_table}")
                    
                    # Collect raw strings from table cells for set comparison later
                    for row in table:
                        row_line = " ".join(cell for cell in row if cell).strip()
                        if row_line:
                            table_contents_set.add(row_line)

            # Extracting text and skipping tables 
            text = page.extract_text()
            if text:
                cleaned_lines = []
                for line in text.splitlines():
                    if not any(line.strip() in row for row in table_contents_set):
                        cleaned_lines.append(line)
                if cleaned_lines:
                    page_content.append("\n".join(cleaned_lines))

            # Extend tables to the list after cleaned text
            page_content.extend(formatted_tables)
            all_text.append("\n".join(page_content))

    pdf_data["data"] = "\n".join(all_text)
    logger.info(f"Successfully Extracted text and table data from {file_name}")
    return pdf_data
   

def inspect_all_pdfs():
    """
    Inspects all PDFs in the PDF directory.
    """
    pdf_dir = paths.paths["pdfs"]
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    pdf_inspections = {}
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_file)
        inspection_result = inspect_pdf(pdf_path)
        pdf_inspections[pdf_file] = inspection_result
    
    return pdf_inspections

def extract_data():
    """
    Extracts data from all PDFs in the Directory
    """
    
    pdf_metadata = inspect_all_pdfs()
    extracted_data = []
    
    for i in pdf_metadata.values():
        if i["has_text"] and not i["has_images"] and not i["has_tables"]:
            extracted_data.append(extract_from_text_pdf(i["pdf_path"])) 
            logger.info(f"Extracted text from {i['pdf_path']} locally")
            
        elif i["has_tables"] and  i["has_text"] and not i["has_images"]:
            extracted_data.append(extract_from_table_pdf(i["pdf_path"]))
            logger.info(f"Extracted text from {i['pdf_path']} locally")

        else :
            extracted_data.append(extract_using_LLM(i["pdf_path"]))
            logger.info(f"Extracted text from {i['pdf_path']} using LLM API")

    return extracted_data


def save_extracted_data_to_file(extracted_data, extracted_text_path):
    """
    Saves the extracted data to a text file.
    """
    with open(extracted_text_path, "w", encoding="utf-8") as f:
        for item in extracted_data:
            f.write(f"File: {item['file_name']}\n")
            f.write(f"Text: {item['data']}\n\n")
    logger.info(f"Extracted data for File : {item['file_name']} & saved to {extracted_text_path}")



def process_and_save_pdf_pipeline():
    """
    End-to-end pipeline to process PDFs and save extracted data.
    """

    extracted_data = extract_data()
    save_extracted_data_to_file(extracted_data, extracted_text_path)
    logger.info(f"PDF Processing complete .")
    

# if __name__ == "__main__":
#     # os.makedirs(os.path.dirname(extracted_text_path), exist_ok=True)
#     # extracted_data = extract_data()
#     # save_extracted_data_to_file(extracted_data, extracted_text_path)
#     # logger.info(f"Extracted data saved to {extracted_text_path}")
#     process_and_save_pdf_pipeline()  # Replace with your PDF file name
    
    
# if __name__ == "__main__":
#     data = extract_data()
#     for item in data:  
#         print(f"File: {item['file_name']}")
#         print(f"Text: {item['data']}...")  
#         print()
        
#     # Save it as a .txt file
#     with open("extracted_data.txt", "w") as f:
#         for item in data:
#             f.write(f"File: {item['file_name']}\n")
#             f.write(f"Text: {item['data']}...\n\n")  
#     print("Data saved to extracted_data.txt")