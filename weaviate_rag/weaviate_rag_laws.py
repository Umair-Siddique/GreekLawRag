import os
import weaviate
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.langchain import LangchainEmbedding
from llama_index.llms.anthropic import Anthropic

from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.retrievers import VectorIndexAutoRetriever  # or .as_retriever
from langchain_voyageai import VoyageAIEmbeddings
import sys
from weaviate.classes.query import Filter
from langchain.prompts import PromptTemplate  
from dotenv import load_dotenv
from google import genai
from google.oauth2 import service_account
from langchain_voyageai import VoyageAIRerank
from dateutil import parser
from datetime import datetime
import re
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)
def extract_keywords_anthropic(query_text, llm):
    prompt = f"""
    You are a legal assistant tasked with extracting strictly law-related keywords from user queries.
    Given the user's query below, return a comma-separated list of exact distinct keywords. Strictly make sure that the keywords relate to the main topic of the user query.
    Also, if any Greek or EU law numbers (e.g., 4855/2021 or 2011/93/ΕΕ) are mentioned in the query, include them in the result.

    Examples:
    Query: "Criminality"
    Keywords: criminality, criminal, criminal law, fraud, homicide

    Query: "Corporate finance laws"
    Keywords: finance, corporate finance laws, corporate governance, securities, compliance, mergers, acquisitions

    Query: "Explain law 4855/2021 on criminal reforms"
    Keywords: criminal reforms, criminal code, law 4855/2021

    Now, your turn:

    Query: "{query_text}"
    Keywords:"""

    response = llm.complete(prompt, max_tokens=60, temperature=0.1)
    keywords_text = response.text.strip()

    # Split the LLM response into a list of keywords
    keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]

    # ✅ Extract Greek/EU law numbers manually using regex
    law_number_matches = re.findall(r'\b\d{3,4}/\d{2,4}\b|\d{4}/\d{4}/ΕΕ', query_text)
    if law_number_matches:
        keywords.extend(law_number_matches)

    return list(set(keywords))  # Remove duplicates

creds = service_account.Credentials.from_service_account_file(
    "credentials/gemini-service-account.json",
    scopes=["https://www.googleapis.com/auth/cloud-platform"]
)

# Initialize genai.Client using Vertex AI
gemini_client = genai.Client(
    credentials=creds,
    vertexai=True,
    project="gemini-api-laws",       # Your real GCP project ID
    location="us-central1"           # Or another valid region like "europe-west4"
)

# 1) Connect (skipping gRPC init checks if needed)
client = weaviate.connect_to_weaviate_cloud(
    cluster_url="https://llqkiruss0myhq2wl8mgta.c0.europe-west3.gcp.weaviate.cloud",
    auth_credentials=weaviate.auth.AuthApiKey(os.getenv("WEAVIATE_API_KEY")),
    skip_init_checks=True,
)

# Wrap vector store
vector_store = WeaviateVectorStore(
    weaviate_client=client,
    index_name="LegalDocuments",
    text_key="text",
    attributes=[
        "legalReferences", "identifier", "title", "url", "gazetteInfo",
        "publicationDate", "effectiveDates", "docType", "status", "primaryDomains",
        "secondaryDomains", "keywords", "regulatoryBodies", "implementingBodies",
        "repealedLegislation", "amendedLegislation", "euReferences", "summary",
    ],embedding_field="content_vector",
)

# Embeddings and LLM
embed_model = LangchainEmbedding(
    VoyageAIEmbeddings(model="voyage-3-large", voyage_api_key=os.getenv("VOYAGE_API_KEY"))
)
llm = Anthropic(
    model="claude-3-7-sonnet-20250219",
    temperature=0.5,
    api_key=os.getenv("CLAUDE_API_KEY")
)

# Index
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model,
    llm=llm
)

# user_query = input("\nPlease enter your legal query: ")

# Directly pass native Weaviate filters (WORKING SOLUTION)


def retrieve_and_rerank(query: str):
    # 1) retrieve and sort by publicationDate descending

    filter_keywords = extract_keywords_anthropic(query, llm)
    print(f"Extracted keywords: {filter_keywords}")

# Create filter conditions for all fields using OR logic
# 1. Conditions for array fields (primaryDomains, secondaryDomains, keywords)
    domain_keyword_conditions = [
     Filter.by_property("primaryDomains").contains_any(filter_keywords),
     # Filter.by_property("secondaryDomains").contains_any(filter_keywords),
     Filter.by_property("keywords").contains_any(filter_keywords)
]

# 2. Conditions for summary field (text search using LIKE)
    summary_conditions = [
     Filter.by_property("summary").like(f"*{kw}*") 
     for kw in filter_keywords
 ]

# Combine all conditions with OR
    all_conditions = domain_keyword_conditions + summary_conditions
    combined_filter = Filter.any_of(all_conditions) if all_conditions else Filter.all()

# Add mandatory status filter with AND
    weaviate_filter = combined_filter & Filter.by_property("status").equal("Ενεργή")


# Retriever with hybrid mode and native Weaviate filtering
    retriever = index.as_retriever(
     vector_store_query_mode="hybrid",
     similarity_top_k=15,     # ← pull only 5
     alpha=0.80,
     use_mmr=True,
     k=15,                    # final MMR size
     vector_store_kwargs={
        "filters": weaviate_filter
    }
 )

    nodes = retriever.retrieve(query)
    nodes = sort_by_pub_date_desc(nodes)

    # 2) metadata fields to include
    metadata_fields = [
        "legalReferences", "identifier", "title", "url", "gazetteInfo",
        "publicationDate", "effectiveDates", "docType", "status", "primaryDomains",
        "secondaryDomains", "keywords", "regulatoryBodies", "implementingBodies",
        "repealedLegislation", "amendedLegislation", "euReferences"
    ]

    # 3) build reranker inputs with full summary
    rerank_inputs = []
    for idx, n in enumerate(nodes):
        props = n.metadata["properties"]

        # grab the full summary
        full_summary = props.get("summary", "")
        if not full_summary.strip():
            print(f"⚠️  Node #{idx} has an EMPTY summary")

        # serialize metadata + complete summary
        lines = [f"{field}: {props.get(field, 'N/A')}" for field in metadata_fields]
        lines.append(f"Summary: {full_summary}")

        rerank_inputs.append("\n".join(lines))

    # 4) filter out any completely empty inputs
    filtered = [(n, txt) for n, txt in zip(nodes, rerank_inputs) if txt.strip()]
    if not filtered:
        return nodes
    filtered_nodes, filtered_texts = zip(*filtered)

    # 5) call the reranker
    reranker = VoyageAIRerank(
        model="rerank-2",
        voyage_api_key=os.getenv("VOYAGE_API_KEY"),
        top_k=10,
        score_threshold=0.7
    )
    resp = reranker._rerank(query=query, documents=list(filtered_texts))

    # 6) map reranked texts back to nodes
    ranked = []
    for result in resp.results:
        for node, original_txt in zip(filtered_nodes, filtered_texts):
            if result.document == original_txt:
                ranked.append(node)
                break

    return ranked




def sort_by_pub_date_desc(nodes):
    def _get_date(n):
        raw = n.metadata["properties"].get("publicationDate")
        if isinstance(raw, datetime):
            return raw
        try:
            return parser.parse(raw)
        except Exception:
            return datetime.min
    return sorted(nodes, key=_get_date, reverse=True)


@app.route('/query', methods=['POST'])
def query_api():
    data = request.get_json(force=True)
    user_query = data.get('query', '')

    # Retrieve and rerank documents
    ranked_nodes = retrieve_and_rerank(user_query)

    # Build prompt contexts with full metadata + summary
    contexts = []
    for i, n in enumerate(ranked_nodes[:5]):
        props = n.metadata['properties']
        block = [
            f"legalReferences: {props.get('legalReferences', 'N/A')}",
            f"identifier: {props.get('identifier', 'N/A')}",
            f"title: {props.get('title', 'N/A')}",
            f"url: {props.get('url', 'N/A')}",
            f"gazetteInfo: {props.get('gazetteInfo', 'N/A')}",
            f"publicationDate: {props.get('publicationDate', 'N/A')}",
            f"effectiveDates: {props.get('effectiveDates', 'N/A')}",
            f"docType: {props.get('docType', 'N/A')}",
            f"status: {props.get('status', 'N/A')}",
            f"primaryDomains: {props.get('primaryDomains', 'N/A')}",
            f"secondaryDomains: {props.get('secondaryDomains', 'N/A')}",
            f"keywords: {props.get('keywords', 'N/A')}",
            f"regulatoryBodies: {props.get('regulatoryBodies', 'N/A')}",
            f"implementingBodies: {props.get('implementingBodies', 'N/A')}",
            f"repealedLegislation: {props.get('repealedLegislation', 'N/A')}",
            f"amendedLegislation: {props.get('amendedLegislation', 'N/A')}",
            f"euReferences: {props.get('euReferences', 'N/A')}",
            f"summary: {props.get('summary', '')}"
        ]
        contexts.append("\n".join(block))
    contexts_str = "\n\n".join([f"Document {i+1}:\n{c}" for i, c in enumerate(contexts)])

    # Format final prompt
    template = """
You are a highly knowledgeable legal research assistant. Use ONLY the provided documents to answer the question below.

When **selecting and ordering** the documents:

* **First**, prioritize the most recent laws sorted by **Publication Date** (newest first).
* **Then**, use the other metadata fields—**legalReferences**, **identifier**, **title**, **gazetteInfo**, **docType**, **status**, **primaryDomains**, **secondaryDomains**, **keywords**, **regulatoryBodies**, **implementingBodies**, **repealedLegislation**, **amendedLegislation**, **euReferences**—to ensure each selected document is the most relevant and accurate for the query.

When you answer:

1. For each law, include:
   * **Publication Date** and **Gazette Info**
   * **Authority**, **Document Type**, and **Status**
   * A **concise, plain-language summary** of its key points and impact
2. Omit full metadata dumps—focus on substance and significance.
3. Immediately after each summary, include its citation link in the format:
   **Citation: <URL>**

QUESTION:
{query}

DOCUMENTS:
{contexts}
"""
    prompt = PromptTemplate(template=template, input_variables=["query", "contexts"])
    full_prompt = prompt.format(query=user_query, contexts=contexts_str)

    # Call Gemini to generate response
    response_chunks = []
    for chunk in gemini_client.models.generate_content_stream(
        model="gemini-2.5-pro-exp-03-25",
        contents=[full_prompt]
    ):
        response_chunks.append(chunk.text)
    final_response = "".join(response_chunks)

    return jsonify({"response": final_response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)



# def print_docs(label, docs):
#     print(f"\n=== {label} ===")
#     for i, n in enumerate(docs, start=1):
#         props = n.metadata["properties"]
#         summary = props.get("summary", "") or ""
#         # truncate to 300 words
#         first_300 = " ".join(summary.split()[:300])
#         print(f"\n-- Document {i} --")
#         print("primaryDomains:  ", props.get("primaryDomains", "N/A"))
#         print("secondaryDomains:  ", props.get("secondaryDomains", "N/A"))
#         print("keywords:               ", props.get("keywords", "N/A"))
#         print("publicationDate:               ", props.get("publicationDate", "N/A"))
#         print("status:               ", props.get("status", "N/A"))
#         print("Law Numbers:               ", [ref["law_number"] for ref in props.get("legalReferences", [])])

#         # print("Structure: ",props)
#         # print(ranked_nodes[0].metadata["properties"])
#         # full_text = ranked_nodes[0].text
#         # print(vars(ranked_nodes[0]))

#         print("Summary (300w):    ", first_300 + ("…" if len(summary.split()) > 300 else ""))
#     print()


# # 5) grab reranked nodes
# ranked_nodes = retrieve_and_rerank(user_query)
# print_docs("Reranked Documents", ranked_nodes)
# # 6) build your prompt contexts exactly as in your sample

# template = """
# You are a highly knowledgeable legal research assistant. Use ONLY the provided documents to answer the question below.

# When **selecting and ordering** the documents:
# - **First**, prioritize the most recent laws sorted by **Publication Date** (newest first).
# - **Then**, use the other metadata fields—**legalReferences**, **identifier**, **title**, **gazetteInfo**, **docType**, **status**, **primaryDomains**, **secondaryDomains**, **keywords**, **regulatoryBodies**, **implementingBodies**, **repealedLegislation**, **amendedLegislation**, **euReferences**—to ensure each selected document is the most relevant and accurate for the query.

# When you answer:
# 1. For each law, include:
#    - **Publication Date** and **Gazette Info**
#    - **Authority**, **Document Type**, and **Status**
#    - A **concise, plain-language summary** of its key points and impact
# 2. Omit full metadata dumps—focus on substance and significance.
# 3. Immediately after each summary, include its citation link in the format:  
#    **Citation: <URL>**

# QUESTION:
# {query}

# DOCUMENTS:
# {contexts}
# """
# prompt = PromptTemplate(template=template, input_variables=["query", "contexts"])

# contexts = "\n\n".join(
#     f"Document {i+1}:\n" +
#     "\n".join([
#         f"Date:                {n.metadata['properties'].get('publicationDate','N/A')}",
#         f"Court:               {n.metadata['properties'].get('court','N/A')}",
#         f"Case Type:           {n.metadata['properties'].get('docType','N/A')}",
#         f"Main Laws:           {n.metadata['properties'].get('main_laws','N/A')}",
#         f"Key Articles:        {n.metadata['properties'].get('key_articles','N/A')}",
#         f"Page URL:            {n.metadata['properties'].get('url','N/A')}",
#     ]) + "\n\nFull Summary:\n" +
#     n.metadata['properties'].get('summary', 'N/A')
#     for i, n in enumerate(ranked_nodes[:5])
# )

# full_prompt = prompt.format(query=user_query, contexts=contexts)

# # 7) stream into Gemini 2.5 Pro
# response_chunks = []
# for chunk in gemini_client.models.generate_content_stream(
#     model="gemini-2.5-pro-exp-03-25",
#     contents=[full_prompt]
# ):
#     print(chunk.text, end="", flush=True)
#     response_chunks.append(chunk.text)

# response = "".join(response_chunks)
# print("\n\n=== Final Response Text ===\n", response)

# client.close()


