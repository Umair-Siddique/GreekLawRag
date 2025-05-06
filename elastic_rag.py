from dotenv import load_dotenv
import os
from elasticsearch import Elasticsearch
from langchain_elasticsearch import ElasticsearchStore
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_voyageai import VoyageAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_voyageai import VoyageAIRerank
from langchain_community.query_constructors.weaviate import WeaviateTranslator
from langchain.chains.query_constructor.base import AttributeInfo
from google import genai
from google.oauth2 import service_account
import anthropic
from langchain.prompts import PromptTemplate  # ensure correct import
# # Sort retrieved documents by date (assuming DD Month YYYY format, adjust parsing as necessary)
from datetime import datetime
import re
from flask import Flask, request, jsonify
from past_cases_rag import process_query_past_cases
import streamlit as st
import sys

load_dotenv()

# Flask initialization
app = Flask(__name__)

# Load API keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY") or st.secrets.get("VOYAGE_API_KEY") 
ANTHROPIC_API_KEY =  os.getenv("CLAUDE_API_KEY") or st.secrets.get("CLAUDE_API_KEY")
ES_URL=os.getenv("ES_URL") or st.secrets.get("ES_URL")
API_KEY= os.getenv("APIKEY_ES_LAWS") 
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
embedding_model = VoyageAIEmbeddings(model="voyage-3", voyage_api_key=VOYAGE_API_KEY)

gemini_creds_dict = {
    "type": os.getenv("GEMINI_TYPE"),
    "project_id": os.getenv("GEMINI_PROJECT_ID"),
    "private_key_id": os.getenv("GEMINI_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GEMINI_PRIVATE_KEY").replace("\\n", "\n"),  # Ensuring proper newlines
    "client_email": os.getenv("GEMINI_CLIENT_EMAIL"),
    "client_id": os.getenv("GEMINI_CLIENT_ID"),
    "auth_uri": os.getenv("GEMINI_AUTH_URI"),
    "token_uri": os.getenv("GEMINI_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("GEMINI_AUTH_PROVIDER_CERT_URL"),
    "client_x509_cert_url": os.getenv("GEMINI_CLIENT_CERT_URL"),
    "universe_domain": os.getenv("GEMINI_UNIVERSE_DOMAIN")
}

creds = service_account.Credentials.from_service_account_info(
    gemini_creds_dict,
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)
# creds = service_account.Credentials.from_service_account_file(
#     "credentials/gemini-service-account.json",
#     scopes=["https://www.googleapis.com/auth/cloud-platform"]
# )

# Initialize genai.Client using Vertex AI
gemini_client = genai.Client(
    credentials=creds,
    vertexai=True,
    project="gemini-api-laws",       
    location="us-central1"        
)

llm = ChatAnthropic(
    model="claude-3-7-sonnet-20250219", # Updated model name based on common availability
    temperature=0.5,
    api_key=ANTHROPIC_API_KEY
)

# Elasticsearch connection
es_client = Elasticsearch(
    ES_URL,
    api_key=API_KEY,
    verify_certs=True
)

# LangChain Elasticsearch vector store initialization
vectorstore = ElasticsearchStore(
    es_connection=es_client,
    index_name="greek_laws_collection",
    query_field="summary", 
    embedding=embedding_model,
    vector_query_field="embedding"  # <--- Add this line, specifying the correct field name
)

# Metadata attributes definition for self-querying
metadata_field_info = [
    AttributeInfo(name="date", description="Publication date of the document in 'DD Month YYYY' format", type="string"),
    AttributeInfo(name="court", description="Name of the court or authority issuing the document", type="string"),
    AttributeInfo(name="case_type", description="Type of legal case or decision", type="string"),
    AttributeInfo(name="primary_issue", description="Main issue addressed by the document", type="string"),
    AttributeInfo(name="legal_field", description="The primary legal field of the document", type="string"),
    AttributeInfo(name="outcome_type", description="Outcome of the legal decision", type="string"),
    AttributeInfo(name="decision_number", description="Decision number of the legal document", type="string"),
    AttributeInfo(name="main_laws", description="Main laws referenced in the document (e.g., Law nos. 4186/2013, 4386/2016)", type="string"),
    AttributeInfo(name="key_articles", description="Key articles referenced in the document (e.g., Article 7 of Law 4186/2013)", type="string"),

    AttributeInfo(name="Page_URL", description="URL of the source web page where the document was found", type="string"),
    AttributeInfo(name="PDF_URL", description="Direct URL to the source PDF document", type="string"),
    AttributeInfo(name="PDF_URL_Saved_To", description="Indicates if/where the PDF was saved ('N/A' if not applicable)", type="string"),
    AttributeInfo(name="monetary_amount", description="Monetary value mentioned, if any ('Δ/Υ' if not applicable/available)", type="string"), # Using string as 'Δ/Υ' is text
    AttributeInfo(name="currency", description="Currency related to the monetary amount ('Δ/Υ' if not applicable/available)", type="string"), # Using string as 'Δ/Υ' is text
    AttributeInfo(name="important_dates", description="List of important dates or date ranges mentioned in the document", type="list[string]"), # Correct type based on example ['2018-2019']
    AttributeInfo(name="procedure_type", description="Type of procedure involved ('Δ/Υ' if not applicable/available)", type="string"), # Using string as 'Δ/Υ' is text
    AttributeInfo(name="court_level", description="The level of the court ('Δ/Υ' if not applicable/available)", type="string"), # Using string as 'Δ/Υ' is text
    AttributeInfo(name="File_Name", description="The original filename of the source document (e.g., PDF filename)", type="string"),
    AttributeInfo(name="summary", description="A generated summary of the document content", type="string"),
    AttributeInfo(name="chunk_id", description="A unique identifier assigned to this specific text chunk or document", type="string"),
]

# Create SelfQueryRetriever with LangChain
self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vectorstore,
    document_contents=["summary"],
    metadata_field_info=metadata_field_info,
    structured_query_translator=WeaviateTranslator(),
    enable_limit=True
)


def extract_year(date_str):
    # Extracts the first 4-digit number found (assuming it's the year)
    match = re.search(r'(\d{4})', date_str)
    if match:
        return int(match.group(1))
    else:
        return 0  # or some default old year if not found

def process_query(user_query):
    query = (
    f'Retrieve the latest and most recent legal documents specifically addressing '
    f'"User query: {user_query}". Prioritize documents from the most recent year available. '
    'Additionally, ensure that the documents meet as many of the following criteria as possible:\n\n'
    '- Court or Authority: [Specify relevant court or administrative body, if applicable]\n'
    '- Case Type: [Clearly state the type of legal decision or case you\'re looking for, e.g., "Decision on criminal cases", "Environmental regulation"]\n'
    '- Primary Issue: [Clearly state the primary legal issue, e.g., "Professional sports regulation", "Hunting bans due to snowfall"]\n'
    '- Legal Field: [Specify the relevant legal domain, e.g., "Criminal Law", "Environmental Law", "Labor Law"]\n'
    '- Outcome Type: [Specify desired outcome, e.g., "Prohibition", "Approval", "Modification"]\n'
    '- Decision Number: [Include decision number if known]\n'
    '- Main Laws Referenced: [List any specific laws you want referenced, e.g., "Law nos. 4186/2013, 4386/2016, 4473/2017"]\n'
    '- Key Articles Referenced: [List specific articles from laws, if relevant, e.g., "Article 7 of Law 4186/2013, Article 66 of Law 4386/2016"]\n\n'
    'Use the metadata fields explicitly provided to filter and retrieve precisely relevant documents.'
)

    retrieved_docs = self_query_retriever.invoke(query)



# Sort by year to ensure most recent is first
    sorted_docs = sorted(
     retrieved_docs,
     key=lambda doc: extract_year(doc.metadata.get('date', '')),
     reverse=True)

    print("=== Retrieved Documents (pre‑rerank) ===")
    for idx, doc in enumerate(sorted_docs, 1):
     date = doc.metadata.get('date', 'N/A')
     url  = doc.metadata.get('Page_URL', 'N/A')
     case_type  = doc.metadata.get('case_type', 'N/A')
     main_laws  = doc.metadata.get('main_laws', 'N/A')
     key_articles  = doc.metadata.get('key_articles', 'N/A')
     court  = doc.metadata.get('court', 'N/A')
     snippet = doc.metadata.get('summary', 'N/A')[:200].replace("\n", " ")  # first 200 chars, single line
     print(f"{idx}. Date: {date}")
     print(f"   case_type: {case_type}")
     print(f"   main_laws: {main_laws}")
     print(f"   key_articles: {key_articles}")
     print(f"   court: {court}")
     print(f"   URL: {url}")
     print(f"   Preview: {snippet}...")
     print("-" * 60)

# --- 1. Build a list of “rich” strings for reranking ---
    rerank_inputs = []
    for doc in sorted_docs:
      md = doc.metadata
      rich_text = (
        f"Date: {md.get('date','N/A')}\n"
        f"Court: {md.get('court','N/A')}\n"
        f"Case Type: {md.get('case_type','N/A')}\n"
        f"Issue: {md.get('primary_issue','N/A')}\n"
        f"Decision Number: {md.get('decision_number','N/A')}\n"
        f"Laws: {md.get('main_laws','N/A')}\n"
        f"Articles: {md.get('key_articles','N/A')}\n"
        f"Summary:\n{doc.metadata.get('summary', 'N/A')}\n"
        f"URL: {md.get('Page_URL','N/A')}"
      )
      rerank_inputs.append(rich_text)

# 2. Rerank using VoyageAIRerank
    reranker = VoyageAIRerank(
     model="rerank-2",
     voyage_api_key=VOYAGE_API_KEY,
     top_k=5
 )
    rerank_response = reranker._rerank(
     query=query,
     documents=rerank_inputs
    )

# 3. Map the reranked strings back to your sorted_docs list
    ranked_docs = []
    for result in rerank_response.results:
      text = result.document.strip()
      for doc, original in zip(sorted_docs, rerank_inputs):
        if original.strip() == text:
            ranked_docs.append(doc)
            break

    template = """
You are a highly knowledgeable legal research assistant. Your task is to provide a detailed, plain-language explanation of how the provided legal documents address the user's query.

When answering, please follow this structure:

1. Clearly explain the relevance of each provided document to the user's query.
2. Identify the most recent laws first (sorted by publication date, newest to oldest).
3. For each law, provide the following clearly and in detail:
   - **Date and Issuing Authority**  
   - **Law Numbers and Key Articles** (explicitly mention them)
   - **Detailed Summary** explaining in plain language the key points, implications, and practical impact of each law, specifically relating to the user's query.
4. After each detailed summary, include a citation link clearly formatted as:  
   **Citation: <Page_URL>**

Be thorough but clear and understandable. Focus explicitly on clarifying the substance and importance of each law and how it addresses the user's specific question.
If the user query and retrieved document are completed irrelevant to each other of different Topic don't mention anyhthing.

QUESTION:
{query}

DOCUMENTS:
{contexts}
"""


    prompt = PromptTemplate(
     template=template,
     input_variables=["query", "contexts"]
    )
# chain = LLMChain(llm=llm, prompt=prompt)

# Build contexts with every metadata field + full summary
    contexts = "\n\n".join(
     # for each of the top‑5 docs
     f"Document {i+1}:\n" +
     "\n".join([
        f"Date:                {doc.metadata.get('date','N/A')}",
        f"Court:               {doc.metadata.get('court','N/A')}",
        f"Case Type:           {doc.metadata.get('case_type','N/A')}",
        f"Issue:               {doc.metadata.get('primary_issue','N/A')}",
        f"Legal Field:         {doc.metadata.get('legal_field','N/A')}",
        f"Outcome Type:        {doc.metadata.get('outcome_type','N/A')}",
        f"Decision Number:     {doc.metadata.get('decision_number','N/A')}",
        f"Main Laws:           {doc.metadata.get('main_laws','N/A')}",
        f"Key Articles:        {doc.metadata.get('key_articles','N/A')}",
        f"Page URL:            {doc.metadata.get('Page_URL','N/A')}",
        f"PDF URL:             {doc.metadata.get('PDF_URL','N/A')}",
        f"Saved PDF To:        {doc.metadata.get('PDF_URL_Saved_To','N/A')}",
        f"Monetary Amount:     {doc.metadata.get('monetary_amount','N/A')}",
        f"Currency:            {doc.metadata.get('currency','N/A')}",
        f"Important Dates:     {doc.metadata.get('important_dates','N/A')}",
        f"Procedure Type:      {doc.metadata.get('procedure_type','N/A')}",
        f"Court Level:         {doc.metadata.get('court_level','N/A')}",
        f"File Name:           {doc.metadata.get('File_Name','N/A')}",
        f"Chunk ID:            {doc.metadata.get('chunk_id','N/A')}",
     ]) +
     "\n\nFull Summary:\n" +
     doc.metadata.get('summary', 'N/A')
     for i, doc in enumerate(ranked_docs[:5])
)
    full_prompt = prompt.format(query=user_query, contexts=contexts)
# Now invoke your chain with the richer contexts:
# Storage for streamed response
    print("\n=== Laws Retrieved (Final Response) ===\n")
    minimal_law_metadata = [{
    "date": doc.metadata.get('date', ''),
    "main_laws": doc.metadata.get('main_laws', ''),
    "key_articles": doc.metadata.get('key_articles', ''),
    "legal_field": doc.metadata.get('legal_field', ''),
    "court": doc.metadata.get('court', ''),
    "decision_number": doc.metadata.get('decision_number', ''),
    "summary": doc.metadata.get('summary', '')[:500],  # brief summary for context
    "Page_URL": doc.metadata.get('Page_URL', ''),
        } for doc in ranked_docs[:5]]

    # response_chunks = []
    # try:
    #     for chunk in gemini_client.models.generate_content_stream(
    #         model="gemini-2.5-pro-exp-03-25",
    #         contents=[full_prompt]
    #     ):
    #         print(chunk.text, end="", flush=True)
    #         response_chunks.append(chunk.text)
        
    #     # Only proceed after successful completion of law retrieval and printing
    #     response = "".join(response_chunks)
    #     print("\n\n=== End of Laws Retrieval ===\n")

    #     # Now explicitly proceed to past court cases retrieval
    #     print("\n=== Retrieving Relevant Past Court Cases ===\n")

    # except Exception as e:
    #     print(f"\n--- Error during law retrieval streaming ---")
    #     print(f"An error occurred: {e}")
    #     partial_text = "".join(response_chunks)
    #     response = f"(Streaming interrupted by error: {e})\n{partial_text}"
    #     print(response)

    # print("\n=== Final Response Text ===\n", response)

    # ── CHANGE HERE ──
    # Return what Streamlit / Flask expect:
    #   1) ranked_docs   (list of top-5 Document objects)
    #   2) full_prompt   (the Gemini prompt – Streamlit streams this)
    #   3) minimal_law_metadata  (for your past-cases RAG step)
    return ranked_docs, full_prompt, minimal_law_metadata

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query')

    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    # Execute your existing process_query logic here
    response, minimal_law_metadata = process_query(user_query)

    # Call past court cases API using minimal law metadata
    past_cases_response_text = process_query_past_cases(user_query, response)

    combined_response = {
        "law_response": response,
        "past_cases_response": past_cases_response_text
    }

    return jsonify(combined_response)


def generate_law_response(full_prompt: str) -> str:
    """Blocking (non-stream) Gemini call – returns the full answer text."""
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-pro-exp-03-25",
        contents=[full_prompt]
    )
    return resp.text


def stream_gemini_answer(full_prompt: str, collector: list | None = None):
    """
    Streaming generator for Streamlit.
    If `collector` list is passed, all chunks are appended to it so you later
    have the full answer without an extra Gemini call.
    """
    for ch in gemini_client.models.generate_content_stream(
            model="gemini-2.5-pro-exp-03-25",
            contents=[full_prompt]):
        if collector is not None:
            collector.append(ch.text)
        yield ch.text


# ─────────────────────────── Flask JSON API ──────────────────────────────────
from past_cases_rag import process_query_past_cases   # keeps same import

app = Flask(__name__)

@app.route("/query", methods=["POST"])
def handle_query():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    docs, full_prompt, law_meta = process_query(user_query)
    law_response = generate_law_response(full_prompt)
    past_cases_resp = process_query_past_cases(user_query, law_response)

    return jsonify({
        "law_response":        law_response,
        "past_cases_response": past_cases_resp
    })


# ─────────────────────────── Streamlit UI ────────────────────────────────────
def streamlit_app():
    st.set_page_config(page_title="Greek Law Assistant", page_icon="⚖️")
    st.title("⚖️ Greek Law Retrieval Assistant")

    user_q = st.text_input("Enter your legal query and hit **Enter**:")
    if not user_q:
        return

    # ── Retrieval phase (progress bar) ───────────────────────────────────────
    with st.status("Retrieving documents …", expanded=False):
        ranked_docs, full_prompt, minimal_meta = process_query(user_q)

    # ── 1. Show retrieved documents first ────────────────────────────────────
    st.header("📄 Top-5 Retrieved Documents")
    for i, doc in enumerate(ranked_docs[:5], 1):
        md = doc.metadata
        with st.expander(f"Document {i}: {md.get('date','N/A')} • {md.get('court','N/A')}"):
            st.write("**Metadata**")
            st.json(md, expanded=False)
            st.write("**Full Summary**")
            st.write(md.get("summary", "N/A"))

    # ── 2. Stream law answer --------------------------------------------------
    st.header("📝 Gemini Answer (streaming)")
    collected_chunks: list[str] = []
    st.write_stream(stream_gemini_answer(full_prompt, collector=collected_chunks))

    full_answer_text = "".join(collected_chunks)     # have the whole answer now

    # ── 3. Past court cases ---------------------------------------------------
    st.header("🏛️ Past Court Cases")
    past_cases_text = process_query_past_cases(user_q, full_answer_text)
    st.write(past_cases_text)

if __name__ == "__main__":
        streamlit_app()
