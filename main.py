from langchain_community.document_loaders import YoutubeLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Step 1a - Indexing (Document Ingestion)

# The full URL of the YouTube video
video_link = "https://www.youtube.com/watch?v=SP-b_G74Nuk&list=WL&index=22"

try:
    # LangChain's YoutubeLoader to fetch the transcript directly from the URL
    # We specify the language as 'en' for English
    loader = YoutubeLoader.from_youtube_url(video_link, add_video_info=False, language="en")
    
    # The .load() method returns a list of Document objects
    # For a single video, this list will usually contain just one document
    documents = loader.load()
    
    # The actual text content is stored in the 'page_content' attribute
    # We access the first document in the list and get its content
    transcript = documents[0].page_content
    
    # Print the fetched transcript to the console
    # print("--- YouTube Transcript ---")
    # print(transcript)
    # print("\n--- End of Transcript ---")

except Exception as e:
    # Catch any potential errors, such as transcripts being disabled
    print(f"An error occurred: {e}")
    print("This could be because transcripts are disabled for this video or the link is invalid.")
    
    
    
#Step 1b - Indexing (Text Splitting)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])
# print(f"{chunks[67].page_content}") 


#Step 1c & 1d - Indexing (Embedding Generation and Storing in Vector Store)

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
print("Generating embeddings with Hugging Face model...")
vector_store = FAISS.from_documents(chunks, embeddings)
print("Vector store created successfully!")
vector_store.index_to_docstore_id

# Step 2 - Retrieval
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
print(retriever.invoke('What is Langgraph?'))



# Step 3 - Augmentation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-latest")
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}
    """,
    input_variables = ['context', 'question']
)