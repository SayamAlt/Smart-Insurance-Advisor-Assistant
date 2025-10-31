import streamlit as st
import os, tempfile, json, warnings, logging
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain.vectorstores.faiss import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from typing import Annotated, List

# Setup
warnings.filterwarnings("ignore")
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
logging.getLogger("streamlit").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

if "secrets" in st.secrets:
    OPENAI_API_KEY = st.secrets["secrets"]["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("Missing OPENAI_API_KEY in environment variables.")
    st.stop()

# Streamlit setup
st.set_page_config(page_title="Insurance Advisor", page_icon="🤖", layout="wide")
st.title("🧠 Real-Time Insurance Advisor")

# Load Knowledge Base
@st.cache_resource
def load_knowledge_base():
    loader = DirectoryLoader("policy_docs/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    return FAISS.from_documents(documents=documents, embedding=embeddings)

vector_db = load_knowledge_base()
base_retriever = vector_db.as_retriever(search_kwargs={"k": 6})

# RAG + memory setup
compressor_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
compressor = LLMChainExtractor.from_llm(llm=compressor_llm)
retriever = ContextualCompressionRetriever(base_retriever=base_retriever, base_compressor=compressor)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Output Schema
class AdvisorInsights(BaseModel):
    advisor_cues: Annotated[List[str], Field(description="Key observations or talking points for the advisor.")]
    suggested_reply: Annotated[str, Field(description="One suggested customer-facing reply.")]
    follow_up_questions: Annotated[List[str], Field(description="Three smart follow-up questions.")]
    policy_references: Annotated[List[str], Field(description="Up to four document-based policy references.")]

# LLM & parser setup
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, api_key=OPENAI_API_KEY)
parser = PydanticOutputParser(pydantic_object=AdvisorInsights)
safe_parser = OutputFixingParser.from_llm(llm=llm, parser=parser)

structured_prompt = PromptTemplate.from_template("""
You are **PolicyAdvisorGPT**, a professional insurance assistant representing **PolicyAdvisor**, 
a Canadian online insurance platform that partners with multiple insurers to provide verified coverage insights.

Your role:
- Help customers understand, compare, and clarify insurance details such as life, term life, critical illness, 
  disability, travel, mortgage, and home insurance - based exclusively on PolicyAdvisor’s verified policy documents.
- All actions, follow-ups, and recommendations must occur **within the PolicyAdvisor ecosystem**.
- The customer should **never be asked to contact their insurance provider directly**.
- Instead, always guide them to **apply or continue their insurance process on PolicyAdvisor’s portal**, 
  where licensed PolicyAdvisor advisors handle everything on their behalf.
- If mentioning insurers (e.g., RBC, Manulife, Wawanesa, Canada Life etc.), frame them as 
  **PolicyAdvisor’s partner insurers**, not external entities.
- The customer’s next step should **always** be to connect through PolicyAdvisor’s app or portal 
  — not take independent actions.
- Always maintain a professional, compliant, and advisory tone consistent with PolicyAdvisor’s brand.

You must:
- Use **only** the retrieved excerpts from PolicyAdvisor’s verified insurance documents (CONTEXT below).
- Never instruct customers to contact insurers or manage coverage themselves. 
- Always redirect them to **apply, compare, or follow up through PolicyAdvisor’s portal** where their licensed advisors and partner insurers handle the next steps.
- If further clarification is needed, direct customers to **PolicyAdvisor’s licensed advisors or partner insurers**.
- Ensure all guidance keeps PolicyAdvisor accountable as the advising entity.
- Avoid speculation, hallucination, or general advice not supported by the retrieved documents.
- Never instruct customers to contact insurers or manage coverage themselves. 
- Always redirect them to **apply, compare, or follow up through PolicyAdvisor’s portal** where their licensed advisors and partner insurers handle the next steps.

---------------------
CONTEXT:
{context}
---------------------

CUSTOMER QUESTION:
{question}

Now, return a structured JSON object in exactly this format:

{{
  "advisor_cues": ["..."],
  "suggested_reply": "...",
  "follow_up_questions": ["...", "...", "..."],
  "policy_references": ["..."]
}}
""").partial(format_instructions=safe_parser.get_format_instructions())

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    memory=memory,
    verbose=False
)

# Initialize Whisper
whisper_model = WhisperModel("small.en", device="cpu", compute_type="float32")

# Session state
if "recording" not in st.session_state:
    st.session_state.recording = False
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# UI Buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("🎙️ Start Recording"):
        st.session_state.recording = True
        st.session_state.transcript = None
with col2:
    if st.button("🛑 Stop & Reset"):
        st.session_state.recording = False
        st.session_state.transcript = None
        st.session_state.chat_history.clear()

# Recording State
if st.session_state.recording:
    audio_file = st.audio_input("Record your customer audio: ", key="customer-audio")

    if audio_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_file.read())
            tmp_path = tmp.name

        segments, _ = whisper_model.transcribe(tmp_path, beam_size=5)
        transcript = " ".join([seg.text.strip() for seg in segments])
        os.remove(tmp_path)

        if transcript.strip():
            st.session_state.transcript = transcript
            st.markdown(f"**🗣️ Transcript:** {transcript}")
        else:
            st.session_state.transcript = None

# Automatically trigger Advisor Insights when transcript available
if st.session_state.transcript:
    with st.spinner("Analyzing customer query and retrieving insights..."):
        response = qa_chain.invoke({"question": st.session_state.transcript})
        parsed_output = safe_parser.parse(response["answer"])

        st.markdown("### 💡 Advisor Cues")
        st.markdown("\n".join([f"- {cue}" for cue in parsed_output.advisor_cues]))

        st.markdown("### 💬 Suggested Reply")
        st.text(parsed_output.suggested_reply)

        st.markdown("### 🤔 Follow-Up Questions")
        st.markdown("\n".join([f"- {q}" for q in parsed_output.follow_up_questions]))

        st.markdown("### 📄 Policy References")
        st.markdown("\n".join([f"- {ref}" for ref in parsed_output.policy_references]))

        st.session_state.chat_history.append({
            "question": st.session_state.transcript,
            "answer": parsed_output.model_dump()
        })

        # Save history
        try:
            with open("chat_history.json", "w") as f:
                json.dump(st.session_state.chat_history, f, indent=2)
        except Exception as e:
            st.error(f"Error saving chat history: {e}")