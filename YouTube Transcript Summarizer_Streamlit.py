# Import necessary libraries
import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter

# Import necessary libraries for text pre_processing
import re
import spacy
import nltk
import wordninja
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import WordNetLemmatizer

# Import necessary libraries for generating text chunks, embeddings and vectorstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

# Import necessary libraries for summarization
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain



# Function for retrieving transcripts
def youtube_transcript(video_id):
    # Retrieving the transcripts
    transcript = YouTubeTranscriptApi.get_transcript(video_id)
    
    # Initializing the text formatter and formatting the transcript
    formatter = TextFormatter()
    transcript_formatted = formatter.format_transcript(transcript)
    
    return transcript_formatted

# Function for text preprocessing
def text_preprocessing(transcript):
    # Removing special characters including newline character '\n'
    transcript = re.sub(r'[,\.!?<>/]', '', transcript)
    
    # Splitting concatenated words
    transcript_cleaned = ' '.join(wordninja.split(transcript))
    
    # Converting the text to lower case
    transcript_cleaned = transcript_cleaned.lower()
    
    # Text lemmatization (Spacy)
    # Initialise spacy 'en' model
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    temp_transcript = nlp(transcript_cleaned)
    transcript_lemmatized = ' '.join([token.lemma_ for token in temp_transcript])
    
    # Spelling correction using TextBlob
    transcript_pre_processed = str(TextBlob(transcript_lemmatized).correct())
    
    return transcript_pre_processed

# Function for chunk of text
def text_chunk(processed_transcript):
    # Creating an instance of RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, 
        chunk_overlap=300, 
        length_function=len)
    
    # Splitting the processed transcript into chunks
    chunks = text_splitter.split_text(processed_transcript)
    
    return chunks

def vectorstore(chunks_transcript):
    # Initializing the HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Creating qdrant vectorstore
    collection_name = "YouTube_Transcript"
    qdrant_vectorstore = Qdrant.from_texts(chunks_transcript,
                               embeddings,
                               location=":memory:",  # Local mode with in-memory storage only
                               collection_name= collection_name)
    
    return qdrant_vectorstore

# Function for generating the summary
def generating_summary(chunks_transcript, llm):
    
    # Defining the template for prompting
    template = """
           Please summarize the following text:
           
           ```
           {text}
           ```
           
           """
    
    # Creating the PromptTemplate object
    prompt = PromptTemplate(template=template, input_variables=["text"])

    # Creating the LLMChain object
    # LLMChain is for chaining tasks
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    
    summaries = []
    for chunk in chunks_transcript:
        summary = llm_chain.run(text=chunk)
        summaries.append(summary)
    
    summarized_text = ' '.join(summaries)
    
    return summarized_text

import streamlit as st

def main():
    null = None
    st.title("YouTube Video Summarizer 🎥")
    
    # Input box for YouTube URL
    youtube_url = st.text_input("Enter YouTube URL")
    
    if st.button('Summarize') and youtube_url:
        
        video_id = youtube_url.split("=")[-1]
        transcript = youtube_transcript(video_id)
        
        # Pre_processing the retreived transcripts
        processed_transcript = text_preprocessing(transcript)
        
        # Generating text chunks from the processed transcript
        chunks_transcript = text_chunk(processed_transcript)
        
        # Creating the Qdrant vector store for the generated embeddings
        #qdrant_client = vectorstore(chunks_transcript)
        summarization_pipeline = pipeline("summarization", 
                                  model="MBZUAI/LaMini-Flan-T5-248M", 
                                  max_length=100,
                                  min_length=25)
        llm = HuggingFacePipeline(pipeline=summarization_pipeline)
        
        transcript_summary = generating_summary(chunks_transcript, llm)
        
        # Display summaryz
        st.subheader("Summary")
        st.write(f"<div style='text-align: justify;'>{transcript_summary}</div>", unsafe_allow_html=True)
    else:
        st.error("Please enter a valid YouTube video URL")
        
if __name__ == "__main__":
    main() 