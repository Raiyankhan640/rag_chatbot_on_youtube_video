import streamlit as st
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables from a .env file
load_dotenv()

# --- Application Configuration ---
st.set_page_config(
    page_title="Video Sage",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)


# --- Core Logic Functions ---

def create_rag_chain(retriever):
    """
    Creates the main Retrieval-Augmented Generation (RAG) chain
    for answering questions based on the video context.
    """
    prompt_template = """
    You are "Video Sage," a friendly and knowledgeable AI assistant.
    Your purpose is to help users understand the content of a YouTube video
    by answering their questions based ONLY on the provided transcript.

    **Instructions:**
    1. Analyze the "Transcript Context" to find the most relevant information for the user's question.
    2. Construct a clear, concise, and helpful response.
    3. If the answer is not found in the transcript, you MUST state:
       "I'm sorry, the video transcript does not seem to cover that topic." Do not make up information.
    4. Respond in the same language as the user's question.

    **Transcript Context:**
    {context}

    **User's Question:**
    {question}

    **Helpful Answer:**
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
    output_parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The fix is here: we explicitly extract the question for the retriever.
    rag_chain = (
        {
            "context": (lambda x: x['question']) | retriever | RunnableLambda(format_docs),
            "question": lambda x: x['question']
        }
        | prompt
        | llm
        | output_parser
    )
    return rag_chain


@st.cache_resource(show_spinner="Processing Video...")
def prepare_vector_store(video_url):
    """
    Loads, splits, and embeds the video transcript to create a retriever.
    This function is cached to avoid reprocessing the same video.
    """
    try:
        # Load transcript
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False, language="en")
        documents = loader.load()
        transcript = documents[0].page_content

        # Split transcript
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.create_documents([transcript])

        # Generate embeddings and create vector store
        embeddings_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

        # Create a temporary directory for FAISS index
        with tempfile.TemporaryDirectory() as temp_dir:
            index_path = os.path.join(temp_dir, "faiss_index")
            vector_store = FAISS.from_documents(chunks, embeddings)
            vector_store.save_local(index_path)
            # Reload to ensure it's ready for use
            reloaded_vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)


        return reloaded_vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    except Exception as e:
        st.error(f"An error occurred while processing the video: {e}")
        return None

# --- Streamlit UI ---

st.title("🤖 Video Sage")
st.markdown("Ask questions about any YouTube video. Just provide the link below to get started.")

# Initialize session state for chat history and chain
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Input for YouTube URL
video_url = st.text_input("Enter YouTube Video URL:", key="video_url_input")

if video_url:
    # Prepare the RAG chain when a new URL is provided
    retriever = prepare_vector_store(video_url)
    if retriever:
        st.session_state.rag_chain = create_rag_chain(retriever)
        st.success("Video processed successfully! You can now ask questions.")
    else:
        st.session_state.rag_chain = None # Reset if processing fails


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about the video"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if the RAG chain is ready
    if st.session_state.rag_chain:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.rag_chain.invoke({"question": prompt})
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.warning("Please enter a valid YouTube URL and wait for it to be processed before asking questions.")

