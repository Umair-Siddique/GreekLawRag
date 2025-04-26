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

load_dotenv()

# Load API keys
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
ANTHROPIC_API_KEY = os.getenv("CLAUDE_API_KEY")
ES_URL= os.getenv("ES_URL")
API_KEY= os.getenv("APIKEY_ES_LAWS")
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
embedding_model = VoyageAIEmbeddings(model="voyage-3", voyage_api_key=VOYAGE_API_KEY)


creds = service_account.Credentials.from_service_account_file(
    "credentials/gemini-service-account.json",
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
You are a highly knowledgeable legal research assistant. Use ONLY the provided documents to answer the question below.

When you answer:
1. Identify the most recent laws, sorted by publication date (newest first).
2. For each law, include only:
   - Date and Authority
   - A concise, plain‑language summary of its key points and impact.
3. Omit detailed metadata listings—focus on explaining the substance and significance of each law.
4. Immediately after each summary, include its citation link in the format: **Citation: <Page_URL> and do include key articles and Main laws if possible**.

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
    full_response_text = []

# Streaming invocation of Claude
    try:
    #   with anthropic_client.messages.create(
    #     model="claude-3-7-sonnet-20250219",
    #     max_tokens=3500,
    #     temperature=0.5,
    #     messages=[{"role": "user", "content": full_prompt}],
    #     stream=True
    #   ) as stream:
    #      print("\n=== Claude Streaming Response ===\n")
    #      for chunk in stream:
    #         if chunk.type == "content_block_delta":
    #             if hasattr(chunk.delta, 'text') and chunk.delta.text:
    #                 print(chunk.delta.text, end="", flush=True)
    #                 full_response_text.append(chunk.delta.text)
    #      print()  # newline after streaming finishes

    #      final_text = "".join(full_response_text)

    #      if final_text:
    #         response = final_text
    #      else:
    #         print("\nWarning: Stream finished but no text content received.")
    #         response = "Warning: Received no text content from Claude stream."
     response_chunks = []
     for chunk in gemini_client.models.generate_content_stream(
      model="gemini-2.5-pro-exp-03-25",
      contents=[full_prompt]
     ):
    # each `chunk` is a small piece of the answer as it’s generated
       print(chunk.text, end="", flush=True)
       response_chunks.append(chunk.text)
     response = "".join(response_chunks)

    except Exception as e:
      print(f"\n--- Error calling Claude API (Streaming) ---")
      print(f"An error occurred: {e}")
      partial_text = "".join(full_response_text)
      if partial_text:
        response = f"(Streaming interrupted by error: {e})\n{partial_text}"
      else:
        response = f"Error processing streaming request: {e}"

# Final output
    print("\n\n=== Final Response Text ===\n", response)

while True:
    user_query = input("\nPlease enter your query ('exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    process_query(user_query)


