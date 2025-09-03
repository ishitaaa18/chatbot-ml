from PyPDF2 import PdfReader
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os

file_path="qp.pdf"

import pytesseract

#only text pdf
def extract_text_from_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

#pure image
def extract_text_from_image(image_path):
    return pytesseract.image_to_string(Image.open(image_path))


#hybrid pdf
def extract_text_hybrid(pdf_path, dpi=300):
    reader = PdfReader(pdf_path)
    full_text = ""
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():  
           
            full_text += text + "\n"
        else:
            
            print(f"OCR on page {page_num+1}")
            images = convert_from_path(pdf_path, dpi=dpi, first_page=page_num+1, last_page=page_num+1)
            for img in images:
                ocr_text = pytesseract.image_to_string(img)
                full_text += ocr_text + "\n"
    
    return full_text


#print(extract_text_from_pdf(file_path))
#print(extract_text_from_image("imagee.png"))
#print(extract_text_hybrid("C2K231153_ShravaniSawant_SE_AMCAT_Result.pdf"))
