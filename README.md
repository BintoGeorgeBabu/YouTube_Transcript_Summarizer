## 📌 Abstractive YouTube Transcript Summarizer

### 🔥 Overview
This project focuses on developing an **Abstractive Summarizer for YouTube Transcripts**. It extracts key insights from YouTube videos using **LaMini-Flan-T5-248M** and **Llama-2-7b-chat-hf**, enabling concise and meaningful summaries from lengthy transcripts.

### ✨ Features
- **YouTube Transcript Extraction**: Retrieves transcripts using the YouTube Transcript API.  
- **Text Preprocessing**: Cleans and normalizes text by removing special characters, correcting spelling, and standardizing formatting.  
- **Chunking Mechanism**: Splits long transcripts into manageable segments using **LangChain Recursive Character Text Splitter**.  
- **Abstractive Summarization**: Utilizes **LaMini-Flan-T5-248M** and **Llama-2-7b-chat-hf** to generate concise summaries.  

---

## 🚀 Methodology

### 1️⃣ Data Collection & Preprocessing  
- Extract transcripts using **YouTube Transcript API**.  
- Clean text by removing noise and normalizing the structure.  

### 2️⃣ Text Chunking  
- Uses **LangChain Recursive Character Text Splitter** to manage token limitations.  
- Implements **chunk overlap** to maintain coherence between text segments.  

### 3️⃣ Summarization Model  
- Employs **LaMini-Flan-T5-248M** and **Llama-2-7b-chat-hf** for abstractive summarization.  
- Evaluates summaries across **different video durations and topics**.  

---

## 📊 Results & Evaluation
- Fine-tuned **LLMs improve summarization accuracy** significantly.  
- **Chunking mechanism enhances coherence**, ensuring high-quality summaries.  

---

## 🛠 Tech Stack
- **Python**  
- **LangChain**  
- **Hugging Face Transformers**  
- **YouTube Transcript API**  
- **PyTorch**  
- **LaMini-Flan-T5 / Llama-2**  

---

## 🏗️ Installation & Usage  

### 📌 Prerequisites  
- Install dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

### 🔄 Usage  
- Extract YouTube transcripts:
  ```python
  from youtube_transcript_api import YouTubeTranscriptApi
  transcript = YouTubeTranscriptApi.get_transcript("VIDEO_ID")
  ```
- Run summarization:
  ```python
  from transformers import pipeline
  summarizer = pipeline("summarization", model="LaMini-Flan-T5-248M")
  summary = summarizer(transcript)
  print(summary)
  ```

## License
This project is licensed under the MIT License.

## Contributing
Pull requests are welcome! Feel free to submit issues or suggestions.


