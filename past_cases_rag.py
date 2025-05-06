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
import re, json
import streamlit as st

load_dotenv()

# Load API keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY") or st.secrets.get("VOYAGE_API_KEY") 
ANTHROPIC_API_KEY =  os.getenv("CLAUDE_API_KEY") or st.secrets.get("CLAUDE_API_KEY")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
embedding_model = VoyageAIEmbeddings(model="voyage-3", voyage_api_key=VOYAGE_API_KEY)
ES_URL = os.getenv("ES_URL") or st.secrets.get("ES_URL")
API_KEY = os.getenv("PASTCASE_APIKEY") or st.secrets.get("PASTCASE_APIKEY")

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

pastCourtVectorStore = ElasticsearchStore(
    es_connection=es_client,
    index_name="pastcase_collection",

    # keyword / BM25 search will run against this text field
    query_field="fullText",

    embedding=embedding_model,

    # dense-vector similarity search will run against this vector field
    vector_query_field="embedding"
)

past_court_decision_meta_info = [
    AttributeInfo(
        name="court",
        description="Court that issued the decision (e.g., «Εφετείο Πειραιώς»)",
        type="string"
    ),
    AttributeInfo(
        name="decision_number",
        description="Official decision number as printed in the source (usually «Ν/Έτος»)",
        type="string"
    ),
    AttributeInfo(
        name="decision_date",
        description="Date the decision was rendered or published, ISO-8601 format «YYYY-MM-DD»",
        type="date"
    ),
    AttributeInfo(
        name="case_type",
        description="Procedural posture / kind of case (e.g., Έφεση, Αντέφεση, Αναίρεση)",
        type="string"
    ),
    AttributeInfo(
        name="main_laws",
        description="Primary statutes or codes referenced (comma-separated Greek titles or abbreviations)",
        type="string"
    ),
    AttributeInfo(
        name="key_articles",
        description="Specific legal articles and paragraphs cited (comma-separated)",
        type="string"
    ),
    AttributeInfo(
        name="primary_issue",
        description="Core substantive or procedural issue adjudicated",
        type="string"
    ),
    AttributeInfo(
        name="monetary_amount",
        description="All money amounts mentioned in the decision text or dispositif",
        type="string"
    ),
    AttributeInfo(
        name="currency",
        description="Currency for the monetary amounts (e.g., EUR)",
        type="string"
    ),
    AttributeInfo(
        name="important_dates",
        description="Other significant litigation or factual dates (comma-separated «YYYY-MM-DD»)",
        type="string"
    ),
    AttributeInfo(
        name="procedure_type",
        description="Procedural track followed (e.g., τακτική διαδικασία, ειδική διαδικασία)",
        type="string"
    ),
    AttributeInfo(
        name="legal_field",
        description="Broad legal domain or category (e.g., Αδικοπραξία, Εμπορικό Δίκαιο)",
        type="string"
    ),
    AttributeInfo(
        name="outcome_type",
        description="Outcome/result classification (e.g., Απόρριψη, Αναβολή, Αποδοχή)",
        type="string"
    ),
    AttributeInfo(
        name="court_level",
        description="Judicial tier (Πρωτοδικείο, Εφετείο, Άρειος Πάγος κ.λπ.)",
        type="string"
    ),
    AttributeInfo(
        name="file_name",
        description="Local or archival file name used during ingestion",
        type="string"
    ),
    AttributeInfo(
        name="page_url",
        description="Canonical web page from which the decision was scraped",
        type="string"
    )
]

past_cases_self_query_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=pastCourtVectorStore,
    document_contents=["summary"],
    metadata_field_info=past_court_decision_meta_info,
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
    
def process_query_past_cases(user_query,law_docs):
    query = f"""
Your task is to convert legal queries and retrieved law metadata into structured filters for finding relevant court cases.  
Focus on these key connections between laws and cases:

**Law Metadata Available:**  
{law_docs}

**User Query Context:**  
{user_query}

**Case Database Schema:**  
- main_laws: Specific law numbers referenced (e.g., "Law 262/2024")  
- key_articles: Articles/sections cited (e.g., "Article 238A ΚΠΔ")  
- legal_field: Matching primary legal domains (e.g., "Criminal Procedure")  
- decision_date: Should align with law's effectiveDates  
- court: Relevant judicial authority  
- outcome_type: Type of legal resolution  

**Required Filters:**  
1. MUST include laws from `main_laws` in retrieved law metadata  
2. SHOULD match articles in `key_articles` when present  
3. CONSIDER `legal_field` alignment with law's primaryDomains  
4. OPTIONAL time filters: decision_date ≥ law publicationDate  

**Output Format:**  
```json
{{
  "filters": {{
    "main_laws": ["<law_number_from_metadata>", ...],
    "key_articles": ["<article_from_metadata>", ...],
    "legal_field": "<primaryDomains_from_law>",
    "decision_date": {{
      "after": "<publicationDate_from_law>"
    }},
    "court": "<court_from_law_metadata>"
  }},
  "relevance_boost": {{
    "main_laws": 3.0,
    "key_articles": 2.5,
    "legal_field": 1.8
  }}
}}
    """

    past_cases_retrieved_docs = past_cases_self_query_retriever.invoke(query)

    # Sort by year to ensure most recent is first
    past_cases_sorted_docs = sorted(
     past_cases_retrieved_docs,
     key=lambda doc: extract_year(doc.metadata.get('decision_date', '')),
     reverse=True)

    # for idx, doc in enumerate(past_cases_sorted_docs, 1):
    #  date = doc.metadata.get('decision_date', 'N/A')
    #  url  = doc.metadata.get('page_url', 'N/A')  # Correct metadata key (case-sensitive)
    #  snippet = doc.metadata.get('Summary', 'N/A')[:500].replace("\n", " ")  # correct field is 'Summary' in your metadata
    #  print(f"{idx}. Date: {date}")
    #  print(f"   URL: {url}")
    #  print(f"   Preview: {snippet}...")
    #  print("-" * 60)

# Define prompt template clearly
    prompt_template = """
You are an expert legal assistant specialized in Greek law. Use ONLY the documents provided below to answer the user's query.

When answering:
- Respond in **plain, easy-to-understand language**, as if explaining to someone without a legal background.
- Mention only the relevant past court decisions from the provided context.
- For each court decision you refer to, **explain each of the following fields clearly**:

1. **Decision Date** — When the court made its decision.
2. **Court** — Which court issued the decision.
3. **Decision Number** — The official number assigned to the decision.
4. **Case Type** — What kind of legal case it was (e.g., civil, criminal).
5. **Main Laws Referenced** — The most important laws the court based its decision on.
6. **Key Articles** — Specific legal articles that were important in the case.
7. **Primary Issue** — What the main legal question or dispute was.
8. **Monetary Amount and Currency** — If money was involved, how much and in what currency.
9. **Important Dates** — Other important dates related to the case (e.g., filing date, hearing dates).
10. **Procedure Type** — What type of legal process was used (e.g., appeal, first-instance trial).
11. **Legal Field** — What area of law the case is about (e.g., family law, contract law).
12. **Outcome Type** — What the court decided overall (e.g., upheld, dismissed).
13. **Court Level** — Whether it was a first-level court, an appeals court, or a supreme court.
14. **Summary of the Decision** — A short, clear explanation of what happened and why.
15. **Citation link** — At the end of each case explanation, provide the link formatted like this: **Citation: <Page_URL>**

Be clear, complete, and stay true to the facts found in the documents. Avoid adding any information that is not in the provided material.

QUESTION:
{query}

DOCUMENTS:
{contexts}
    """


    prompt = PromptTemplate(
     template=prompt_template,
     input_variables=["query", "contexts"]
    )

# Build contexts using all metadata fields provided in your actual code
    contexts = "\n\n".join(
     f"Document {i+1}:\n" +
     "\n".join([
        f"Decision Date:      {doc.metadata.get('decision_date', 'N/A')}",
        f"Court:              {doc.metadata.get('court', 'N/A')}",
        f"Decision Number:    {doc.metadata.get('decision_number', 'N/A')}",
        f"Case Type:          {doc.metadata.get('case_type', 'N/A')}",
        f"Main Laws:          {doc.metadata.get('main_laws', 'N/A')}",
        f"Key Articles:       {doc.metadata.get('key_articles', 'N/A')}",
        f"Primary Issue:      {doc.metadata.get('primary_issue', 'N/A')}",
        f"Monetary Amount:    {doc.metadata.get('monetary_amount', 'N/A')}",
        f"Currency:           {doc.metadata.get('currency', 'N/A')}",
        f"Important Dates:    {doc.metadata.get('important_dates', 'N/A')}",
        f"Procedure Type:     {doc.metadata.get('procedure_type', 'N/A')}",
        f"Legal Field:        {doc.metadata.get('legal_field', 'N/A')}",
        f"Outcome Type:       {doc.metadata.get('outcome_type', 'N/A')}",
        f"Court Level:        {doc.metadata.get('court_level', 'N/A')}",
        f"Page URL:           {doc.metadata.get('page_url', 'N/A')}",
     ]) +
     "\n\nFull Summary:\n" +
     doc.metadata.get('summary', 'N/A')
     for i, doc in enumerate(past_cases_sorted_docs[:5])
)

# Generate full prompt
    full_prompt = prompt.format(query=user_query, contexts=contexts)

# Streaming invocation with Gemini 2.5 Pro
    response_chunks = []
    try:
     for chunk in gemini_client.models.generate_content_stream(
        model="gemini-2.5-pro-exp-03-25",
        contents=[full_prompt]
     ):
        print(chunk.text, end="", flush=True)
        response_chunks.append(chunk.text)
     response = "".join(response_chunks)

    except Exception as e:
      print(f"\n--- Error calling Gemini API (Streaming) ---")
      print(f"An error occurred: {e}")
      partial_text = "".join(response_chunks)
      if partial_text:
        response = f"(Streaming interrupted by error: {e})\n{partial_text}"
      else:
        response = f"Error processing streaming request: {e}"

# Final output
    print("\n\n=== Final Response Text ===\n", response)
    return response
