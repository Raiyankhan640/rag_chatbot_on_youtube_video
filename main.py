from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
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
# print(retriever.invoke('What is Langgraph?'))



# Step 3 - Augmentation
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

prompt = PromptTemplate(
    template = """
    You are a helpful and knowledgeable assistant named "Video Sage," designed to answer questions about YouTube videos.

    **Your Task:**
    Based *only* on the provided transcript from the video, provide a comprehensive and helpful answer to the user's question. 
    While you should not use outside knowledge, you can elaborate on the topics mentioned in the transcript to provide a full explanation, as long as it remains true to the video's content.

    **Guidelines:**
    1.  Carefully analyze the "Transcript Context" to find the most relevant information.
    2.  Formulate a clear, elaborate, and helpful response that maintains the main concepts from the video.
    3.  If the transcript doesn't contain the answer, you MUST state: "I'm sorry, but the video transcript doesn't seem to cover that topic." Do not guess or infer information not present.
    4.  Always respond in the same language as the user's question.

    **Transcript Context:**
    {context}

    **User's Question:**
    {question}

    **Your Answer:**
    """,
    input_variables = ['context', 'question']
)

# Sample Question to Test the System
question = "What do I need to know to get started with Langgraph? How can I start making projects with it?"
# Retrieve relevant documents from the vector store
retrieved_docs = retriever.invoke(question)
# Combine the retrieved document contents into a single context string
context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# Final prompt by filling in the template with context and question
final_prompt = prompt.invoke({"context": context_text, "question": question})


# Step 4 - Generation
# response = llm.invoke(final_prompt)
# print("LLM Response:")
# print(response.text)



# Building in Chain
def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

# Create a parallel runnable to handle both context retrieval and question passing
parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

# Create a string output parser
parser = StrOutputParser()
# Connect the parser to the main chain
main_chain = parallel_chain | prompt | llm | parser
print(main_chain.invoke('Can you summarize the video'))