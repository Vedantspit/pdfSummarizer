
import fitz  # PyMuPDF
import openai
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RunnableSequence
import os

openai.api_key = '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'
os.environ['OPENAI_API_KEY'] = openai.api_key


# Function to extract text from PDF using PyMuPDF
def extract_text_from_pdf(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ''
    for page_num in range(pdf_document.page_count):
        page = pdf_document[page_num]
        text += page.get_text()
    return text

# Function to summarize text using LangChain and OpenAI API
def summarize_text(text):
    # Define the OpenAI language model
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.7, openai_api_key=os.environ['OPENAI_API_KEY'])

    # Define a prompt template
    prompt_template = PromptTemplate(template="Summarize the following text: {text}", input_variables=["text"])

    # Define the RunnableSequence
    chain = prompt_template | llm

    # Execute the chain to get the summary
    summary = chain.invoke({"text": text})
    
    return summary

# Path to your PDF file
pdf_path = 'AICAN_JD_ (2).pdf'

# Extract text from the PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Summarize the extracted text
summary = summarize_text(pdf_text)

# Print the summary
print("Summary:")
print(summary)
