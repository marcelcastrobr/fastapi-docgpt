from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from typing import Any
import logging
from dotenv import load_dotenv, find_dotenv

# Import the necessary functions from utils.py
from utils import process_pdf, send_to_qdrant, qdrant_client, qa_ret, get_callback_handler, get_embedding_model, process_pdf_tables


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger  = logging.getLogger('backend')

app = FastAPI()

# Keep tracks of user specific variables, set in login function
session = {} # CHANGED TO SE flask_session instead, extention to handle server side sessions
# Hold agents for users,
# Key-value pairs are the username and AgentExecutor object with the user specific agent
agents = {}

# Frontend URL
FRONTEND_URL = os.getenv("FRONTEND_URL") 

# Loading environment variables
load_dotenv(find_dotenv())


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", FRONTEND_URL],  # Allow requests from your React app (adjust domain if necessary)
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allow all headers
)


class QuestionRequest(BaseModel):
    question: str
    username: str
    country: str
    role: str



# Define a prompt for the API
class RAGChatPromptTemplate(ChatPromptTemplate):
    prompt:str
    
    def __init__(self, template:str):
        self.prompt = ChatPromptTemplate.from_template(template)


# Endpoint to upload a PDF and process it, sending to Qdrant
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file, process it, and store in the vector DB.
    """
    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(file.file.read())
            temp_file_path = temp_file.name

        # Process the PDF to get document chunks and embeddings
        document_chunks = process_pdf(temp_file_path)
        # Process the PDF with tables to get document chunks and embeddings
        #document_chunks = process_pdf_tables(temp_file_path)

        # Get the embedding model           
        embedding_model = get_embedding_model()


        # Send the document chunks (with embeddings) to Qdrant
        success = send_to_qdrant(document_chunks, embedding_model)

        # Remove the temporary file after processing
        os.remove(temp_file_path)

        if success:
            return {"message": "PDF successfully processed and stored in vector DB"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store PDF in vector DB")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

# Endpoint to ask a question and retrieve the answer from the vector DB
@app.post("/ask-question/")
async def ask_question(question_request: QuestionRequest):
    """
    Endpoint to ask a question and retrieve a response from the stored document content.
    """
    try:
        # Retrieve the Qdrant vector store (assuming qdrant_client() gives you access to it)
        qdrant_store = qdrant_client()

        # Get the question from the request body
        question = question_request.question

        # Use the question-answer retrieval function to get the response
        response = qa_ret(qdrant_store, question)

        return {"answer": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")

# A simple health check endpoint
@app.get("/")
async def health_check():
    return {"status": "Success"}


# Endpoint to ask question and retrive answer to frontend streamlit
@app.post("/message")
async def ask_question_2(data: QuestionRequest = Body(None)):    
    """
    Endpoint to ask a question and retrieve a response from the stored document content.
    """

    logger.debug(session)
    if 'item' not in session:
        session['item'] = ''
    
    #if not isinstance(data, dict):
    #    return "Could not handle request, data is not a dictionary"
    
    logger.debug(f"Data: {data}")
    
    try:  
        #response = agents[data['username']](data['input'])
        # Retrieve the Qdrant vector store (assuming qdrant_client() gives you access to it)
        qdrant_store = qdrant_client()

        # Get the question from the request body
        #question = question_request.question
        question = data.question



        # Use the question-answer retrieval function to get the response
        response = qa_ret(qdrant_store, question)

        data = {'ai_response':response, 'source_doc':'', 'pages':''}

        return data
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve answer: {str(e)}")

