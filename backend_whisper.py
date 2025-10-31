import streamlit as st
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode
import numpy as np
import librosa, scipy.io.wavfile as wavfile
import openai, threading, os, tempfile, warnings, logging, json, re
warnings.filterwarnings("ignore") 
from langchain.vectorstores.faiss import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Annotated, List
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

warnings.filterwarnings("ignore")
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
logging.getLogger("streamlit_webrtc").setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Initialize OpenAI client
client = OpenAI()
    
# Streamlit UI Setup
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

# Compressor LLM
compressor_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, api_key=OPENAI_API_KEY)
compressor = LLMChainExtractor.from_llm(llm=compressor_llm)

# Contextual Compression Retriever
retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)

# Define Output Schema
class AdvisorInsights(BaseModel):
    advisor_cues: Annotated[List[str], Field(description="Key observations or talking points for the advisor.")]
    suggested_reply: Annotated[str, Field(description="One suggested customer-facing reply.")]
    follow_up_questions: Annotated[List[str], Field(description="Three smart follow-up questions.")]
    policy_references: Annotated[List[str], Field(description="Up to four document-based policy references.")]
                
# RAG Chain Setup
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.5, api_key=OPENAI_API_KEY)

# Output Parser
parser = PydanticOutputParser(pydantic_object=AdvisorInsights)

safe_parser = OutputFixingParser.from_llm(llm=llm, parser=parser)


# Build the structured prompt
structured_prompt = PromptTemplate.from_template("""
You are **PolicyAdvisorGPT**, a professional Canadian insurance assistant trained exclusively on PolicyAdvisor’s verified insurance documents.

Your only goal is to help users **understand, compare, or clarify** insurance policies — such as life, term life, critical illness, disability, travel, mortgage, and health insurance.

You must:
- Use **only** the retrieved policy excerpts provided in CONTEXT below.
- If the context does **not** mention relevant insurance details, reply: 
  `"Information not found in policy documents."`
- Never include generic or non-insurance advice (e.g. safe driving, weather tips, personal habits).
- Be factual, compliant, and concise.
- Do not hallucinate.
- The transcript may contain disfluencies; focus on the core question.
- Adhere strictly to the output schema provided.

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

Guidelines:
- **advisor_cues**: Extract 2–4 key insurance-related insights drawn from the context.
- **suggested_reply**: Provide a short, professional, factual answer referring to policy coverage or conditions.
- **follow_up_questions**: Ask specific insurance-related follow-up questions (e.g. coverage amount, term length, beneficiaries).
- **policy_references**: Include document names, sections, or clauses mentioned in the context.

If the question or context does not match any policy content, respond with:
{{
  "advisor_cues": ["Information not found in policy documents."],
  "suggested_reply": "Information not found in policy documents.",
  "follow_up_questions": [],
  "policy_references": []
}}
""").partial(format_instructions=safe_parser.get_format_instructions())

# Create the RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=False,
    memory=memory,
    verbose=False
)

# Audio Processor
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = []
        self.lock = threading.Lock()
        self.transcript = ""

    def recv(self, frame):
        audio = frame.to_ndarray().flatten().astype(np.float32)
        with self.lock:
            self.audio_buffer.append(audio)
        return frame

    def get_transcript(self):
        with self.lock:
            if not self.audio_buffer:
                return ""

            # Combine buffered audio
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            self.audio_buffer.clear()
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Resample to 16kHz (Whisper requirement)
            audio_resampled = librosa.resample(audio_data, orig_sr=48000, target_sr=16000, res_type='kaiser_best')
            
            # Convert to int16 and save as temporary WAV
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wavfile.write(tmp.name, 16000, audio_int16)
                tmp_path = tmp.name

            # Transcribe using OpenAI Whisper API
            try:
                with open(tmp_path, "rb") as f:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language="en"
                    )

                text = getattr(transcript, "text", "").strip() if getattr(transcript, "text", "") else ""
                if text:
                    self.transcript += " " + text
            except Exception as e:
                print("ASR error:", e)
            finally:
                os.remove(tmp_path)

            # Clean up transcript: remove emojis, non-ASCII characters, multiple spaces, etc.
            cleaned_text = re.sub(f"[^\x00-\x7F]+", " ", self.transcript)
            cleaned_text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", cleaned_text)
            cleaned_text = re.sub(f"\s+", " ", cleaned_text).strip()
            return cleaned_text

# WebRTC Streamer
webrtc_ctx = webrtc_streamer(
    key="customer-audio",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=8192,
    media_stream_constraints={"audio": True, "video": False},
    audio_processor_factory=AudioProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
if "memory" not in st.session_state:
    st.session_state.memory = memory
else:
    memory = st.session_state.memory
    
# Transcription & RAG Retrieval
if webrtc_ctx and webrtc_ctx.audio_processor:
    st.info("🎤 Please speak. When finished, click below to generate advisor insights.")

    if st.button("🧩 Get Insights"):
        transcript = webrtc_ctx.audio_processor.get_transcript()
        st.write("🗣️ **Transcript:**", transcript if transcript else "_(No audio detected)_")

        if transcript:
            with st.spinner("Analyzing customer context and retrieving insights..."):
                # Ensure chat history is always defined
                stored_history = st.session_state.get("chat_history", [])
                
                # Convert chat history to list of tuples
                # chat_history = []
                
                # for item in stored_history:
                #     if isinstance(item, dict):
                #         q = item.get("question", "")
                #         a = item.get("answer", "")
                #         if isinstance(a, dict):  # convert dict answer to string summary
                #             a = a.get("suggested_reply", json.dumps(a))
                #         chat_history.append((q, a))
                #     elif isinstance(item, tuple):
                #         chat_history.append(item)
                
                # # Invoke RAG chain
                # response = qa_chain.invoke({
                #     "question": transcript,
                #     "chat_history": chat_history
                # })
                
                response = qa_chain.invoke({
                    "question": transcript
                })
                
                print("RAG Response:", response)
                
                # Parse structured output
                parsed_output = safe_parser.parse(response['answer'])
                
                if isinstance(response, AdvisorInsights):
                    response = response['answer']
                    advisor_cues = "\n".join(f"{cue}" for cue in parsed_output.advisor_cues)
                    follow_up_questions = [f"{q}" for q in parsed_output.follow_up_questions if q.strip()]
                    suggested_reply = "\n" + str(parsed_output.suggested_reply)
                    st.write("**Advisor Cues:**")
                    st.markdown("\n".join([f"- {cue}" for cue in parsed_output.advisor_cues]))
                    st.write("**Suggested Reply:**")
                    st.text(suggested_reply)
                    st.markdown("**Follow-Up Questions:**")
                    st.markdown("\n".join([f"- {q}" for q in follow_up_questions]))
                else:
                    advisor_cues = "\n".join(f"{cue}" for cue in parsed_output.advisor_cues)
                    follow_up_questions = [f"{q}" for q in parsed_output.follow_up_questions if q.strip()]
                    suggested_reply = "\n" + str(parsed_output.suggested_reply) 
                    st.write("**Advisor Cues:**")
                    st.markdown("\n".join([f"- {cue}" for cue in parsed_output.advisor_cues]))
                    st.write("**Suggested Reply:**")
                    st.text(suggested_reply)
                    st.markdown("**Follow-Up Questions:**")
                    st.markdown("\n".join([f"- {q}" for q in follow_up_questions]))

                # Update chat history
                st.session_state.chat_history.append({
                    "question": transcript,
                    "answer": parsed_output.model_dump()
                })

                # Save chat history
                try:
                    serializable_history = [
                        {"question": h["question"], "answer": h["answer"]}
                        for h in st.session_state.chat_history
                    ]
                    with open("chat_history.json", "w") as f:
                        json.dump(serializable_history, f, indent=2)
                except Exception as e:
                    st.error(f"Error saving chat history: {e}")
        else:
            st.warning("No transcript available yet. Please speak into the microphone.")