from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import AzureChatOpenAI
import logging
import os
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler
from unstructured.staging.base import elements_from_base64_gzipped_json
from langchain_community.document_loaders import UnstructuredPDFLoader





# Create a logger for this module
logger = logging.getLogger(__name__)


# Load environment variables (if needed)
load_dotenv()

# API keys and URLs from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")


# Function to process PDF with tables and split it into chunks
def process_pdf_tables(pdf_path):
    """Process the PDF, split it into chunks, and return the chunks."""
    #loader = PyPDFLoader(pdf_path)
    loader = UnstructuredPDFLoader(pdf_path,  
                               mode="elements", 
                               strategy="hi_res",
                               extract_image_block_types=["Image", "Table"],
                               extract_image_block_to_payload = True, 
                               chunking_strategy="by_title", 
                               max_characters=4000,  
                               new_after_n_chars=3800
                               )
    pages = loader.load()
    document_text = "".join([page.page_content for page in pages])


    #Add tables to the document text
    tables = []
    for doc in pages:
        if 'orig_elements' in doc.metadata:
            for orig_element in elements_from_base64_gzipped_json(doc.metadata["orig_elements"]):
                if orig_element.category == "Table" :
                    print(orig_element)
                    tables.append(str(orig_element))
    # Join all table elements into a single string
    document_table = "\n".join(tables)
    print(document_table)
    
    document_all = "\n".join([document_text, document_table])

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,  # Adjust as needed
        chunk_overlap=200  # Adjust as needed
    )
    chunks = text_splitter.create_documents([document_all])

    return chunks

# Function to process PDF and split it into chunks
def process_pdf(pdf_path):
    """Process the PDF, split it into chunks, and return the chunks."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    document_text = "".join([page.page_content for page in pages])

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Adjust as needed
        chunk_overlap=40  # Adjust as needed
    )
    chunks = text_splitter.create_documents([document_text])

    return chunks


# Function to send document chunks (with embeddings) to the Qdrant vector database
def send_to_qdrant(documents, embedding_model):
    """Send the document chunks to the Qdrant vector database."""
    try:
        qdrant = Qdrant.from_documents(
            documents,
            embedding_model,
            url=QDRANT_URL,
            prefer_grpc=False,
            api_key=QDRANT_API_KEY,
            collection_name="xeven_chatbot",  # Replace with your collection name
            force_recreate=True  # Create a fresh collection every time
        )
        return True
    except Exception as ex:
        print(f"Failed to store data in the vector DB: {str(ex)}")
        return False


# Function to initialize the Qdrant client and return the vector store object
def qdrant_client():
    """Initialize Qdrant client and return the vector store."""
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_openai_api_key:
            print("Use Azure OpenAI API")
            # Create the embedding model for Azure OpenAI
            embedding_model = AzureOpenAIEmbeddings(
                openai_api_key=azure_openai_api_key,
                model="text-embedding-ada-002",
                deployment=os.getenv("EMBEDDING"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
    else:
        embedding_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")
    
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    qdrant_store = Qdrant(
        client=qdrant_client,
        collection_name="xeven_chatbot",
        embeddings=embedding_model
    )
    return qdrant_store


# Function to handle question answering using the Qdrant vector store and GPT
def qa_ret(qdrant_store, input_query):
    """Retrieve relevant documents and generate a response from the AI model."""
    try:
        template = """
        Instructions:
            You are trained to extract answers from the given Context and the User's Question. Your response must be based on semantic understanding, which means even if the wording is not an exact match, infer the closest possible meaning from the Context. 

            Key Points to Follow:
            - **Precise Answer Length**: The answer must be between a minimum of 40 words and a maximum of 100 words.
            - **Strict Answering Rules**: Do not include any unnecessary text. The answer should be concise and focused directly on the question.
            - **Professional Language**: Do not use any abusive or prohibited language. Always respond in a polite and gentle tone.
            - **No Personal Information Requests**: Do not ask for personal information from the user at any point.
            - **Concise & Understandable**: Provide the most concise, clear, and understandable answer possible.
            - **Semantic Similarity**: If exact wording isn’t available in the Context, use your semantic understanding to infer the answer. If there are semantically related phrases, use them to generate a precise response. Use natural language understanding to interpret closely related words or concepts.
            - **Unavailable Information**: If the answer is genuinely not found in the Context, politely apologize and inform the user that the specific information is not available in the provided context.

            Context:
            {context}

            **User's Question:** {question}

            Respond in a polite, professional, and concise manner.
        """
        prompt = ChatPromptTemplate.from_template(template)
        retriever = qdrant_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        # Langfuse callback
        user_id = f"qdrant"
        langfuse_handler = get_callback_handler(user_id)

        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        ).with_config({"callbacks": [langfuse_handler]})

        # Get LLM model
        model = get_llm_model()

        output_parser = StrOutputParser()

        rag_chain = setup_and_retrieval | prompt | model | output_parser
        response = rag_chain.invoke(input_query, config={"callbacks": [langfuse_handler]})
        return response

    except Exception as ex:
        return f"Error: {str(ex)}"


# Function that return langfuse callback handler
def get_callback_handler(username):

    try:
        logger.debug(f"Landfuse: getting public key {os.environ['LANGFUSE_PUBLIC_KEY']} and host: {os.environ['LANGFUSE_HOST']}")
        langfuse_handler = CallbackHandler(user_id=username)
        langfuse_handler.auth_check()
        return langfuse_handler
    except KeyError as e:
        logger.error(f"Environment variable {e} not found")
        return ""
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return ""

# Get embedding_model to be used
def get_embedding_model():
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_openai_api_key:
            print("Use Azure OpenAI API")
            # Create the embedding model for Azure OpenAI
            embedding_model = AzureOpenAIEmbeddings(
                openai_api_key=azure_openai_api_key,
                model="text-embedding-ada-002",
                deployment=os.getenv("EMBEDDING"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            )
    else:
        print("Use OpenAI API")
        # Create the embedding model for OpenAI
        embedding_model = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-ada-002"
        )
    return embedding_model


# Get llm model to be used
def get_llm_model():
    azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if azure_openai_api_key:
            print("Use Azure OpenAI API")
            # Create the embedding model for Azure OpenAI
            model = AzureChatOpenAI(
                temperature=0,
                openai_api_key=azure_openai_api_key,
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment=os.getenv("LLM"),
                api_version="2024-05-01-preview",
                model=os.getenv("LLM"),
            )
    else:
        print("Use OpenAI API")
        # Create the embedding model for OpenAI
        model = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=150
        )
    return model
