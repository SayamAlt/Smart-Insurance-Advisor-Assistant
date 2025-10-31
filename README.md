# Smart Insurance Advisor CoPilot

## Overview

Successfully developed a **real-time AI-powered Insurance Advisor CoPilot** that converts customer audio into actionable insights, guiding users effortlessly through insurance coverage. Leveraging advanced NLP and speech-to-text technologies, it delivers instant, accurate recommendations while ensuring all actions are fully managed by PolicyAdvisor and its trusted insurance partners. This AI assistant redefines digital insurance engagement, blending cutting-edge AI with human-centric advisory workflows.

## Features

- **Real-Time Audio Processing:** Capture and transcribe customer audio instantly.
- **Intelligent Policy Guidance:** Provide accurate insurance advice based on PolicyAdvisor’s client policies (RBC, Manulife, Wawanesa, Canada Life, etc.).
- **Action-Oriented Recommendations:** Suggest applying for insurance coverage directly through the PolicyAdvisor portal, removing any manual steps for the customer.
- **Advisor CoPilot Support:** Enhance insurance advisors’ efficiency by automating routine inquiries and providing context-aware suggestions.
- **Policy Reference Integration:** Leverage relevant policy documents and RAG-based retrieval to inform advice.

## Tech Stack

- **Python** for backend processing and orchestration.
- **FastWhisper / Whisper / Faster-Whisper** for speech-to-text transcription.
- **LangChain & LangGraph** for reasoning, knowledge retrieval, and AI workflow orchestration.
- **FAISS** for vector database management and semantic search.
- **Streamlit + streamlit-webrtc** for real-time web interface and audio streaming.
- **AI/ML Libraries:** NumPy, Pandas, Scikit-learn, PyTorch (optional for embeddings and models).

## Installation

```bash
git clone https://github.com/sayamalt/smart-insurance-advisor-assistant.git
cd smart-insurance-advisor-assistant
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:

   ```bash
   streamlit run backend_fastwhisper.py
   ```

2.	Use the web interface to record customer audio.
3.	Receive real-time transcription and AI-driven insurance advice.
4.	Apply for insurance coverage directly through the PolicyAdvisor portal via the assistant.

## Contribution

Contributions are welcome! Please open issues or pull requests for bug fixes, improvements, or new features.

## License

This project is licensed under the Apache License 2.0. See the LICENSE￼file for details.
