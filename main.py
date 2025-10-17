from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Part 1: The RAG Chain  ---

# This function encapsulates the logic for creating the RAG chain
def create_rag_chain(retriever):
    """Creates the main RAG chain for answering questions."""

    # We use the refined prompt from our discussion
    prompt = PromptTemplate(
        template="""
        You are "Video Sage," a friendly and helpful AI assistant designed to help users understand the content of a YouTube video.

        Your main goal is to answer the user's questions based *only* on the provided transcript context. Think of yourself as a guide to the video's content.

        **Guidelines:**
        1.  Carefully read the provided "Transcript Context" to find the most relevant information for the user's question.
        2.  Formulate a clear and helpful response. You can synthesize information from different parts of the context to give a complete answer.
        3.  If the transcript doesn't contain the answer, you MUST state: "I'm sorry, but the video transcript doesn't seem to cover that topic." Do not guess.
        4.  Always respond in the same language as the user's question.

        **Transcript Context:**
        {context}

        **User's Question:**
        {question}

        **Helpful Answer:**
        """,
        input_variables=['context', 'question']
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | parser
    )
    
    return rag_chain

# --- Part 2: The Indexing Logic ---

# This function handles the expensive, one-time setup for a video
def prepare_vector_store(video_url):
    """Loads, splits, and embeds the video transcript to create a retriever."""
    try:
        print("Loading video transcript...")
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False, language="en")
        documents = loader.load()
        transcript = documents[0].page_content

        print("Splitting transcript into chunks...")
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.create_documents([transcript])

        print("Generating embeddings and creating vector store... (This might take a moment)")
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        print("Video processing complete!")
        return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# --- Part 3: The Main Chatbot Loop ---

def main():
    """The main function to run the interactive chatbot."""
    
    # Step 1: Get the video URL from the user
    video_link = input("Please enter the YouTube video URL you want to chat with: ")
    
    # Step 2: Index the video (load, split, embed)
    retriever = prepare_vector_store(video_link)
    
    if retriever is None:
        print("Could not process the video. Please check the URL and try again.")
        return

    # Step 3: Create the RAG chain with the video-specific retriever
    main_chain = create_rag_chain(retriever)
    
    print("\n✅ Video Sage is ready! Ask any questions about the video.")
    print("   Type 'quit' or 'exit' to end the chat.\n")
    
    # Step 4: Start the interactive question-answering loop
    while True:
        question = input("Your Question: ")
        if question.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        # Invoke the chain and print the response
        response = main_chain.invoke(question)
        print("\nVideo Sage:", response, "\n")

# This makes the script runnable
if __name__ == "__main__":
    main()