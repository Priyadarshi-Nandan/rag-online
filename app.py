import os
import time
import chromadb
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from chunking import chunk_text


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_PATH = os.path.join("docs", "solar_system.pdf")
CHUNK_SIZE = 500

# Small batches keep memory low. This came from the memory test:
# 100 chunks at once peaked at about 496 MB, 4 at a time about 279 MB.
BATCH_SIZE = 4

# Only the best few chunks are sent to the LLM. Your tests showed that
# 3 chunks gave the same score as sending everything, with about half
# the tokens.
MAX_CHUNKS_TO_LLM = 3

# Very common words. They are left out of the keyword search because
# they match almost every chunk and only add noise.
STOP_WORDS = [
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "in", "on",
    "at", "to", "for", "and", "or", "with", "by", "from", "as", "it", "its",
    "this", "that", "which", "what", "who", "how", "does", "do", "did",
    "me", "tell", "about"
]

REFUSAL_PHRASES = [
    "i don't have enough information",
    "does not contain",
    "do not contain",
    "doesn't contain",
    "no information about"
]

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Everything the search needs is kept in this dictionary, in memory.
store = {}
store["chunks"] = []
store["by_id"] = {}
store["collection"] = None
store["bm25"] = None


# ============================================================
# READING AND LOADING THE SAMPLE DOCUMENT
# ============================================================

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text is None:
            page_text = ""
        text = text + page_text + "\n"
    return text


def tokenize(text):
    # Turns text into a list of lowercase words for the keyword search.
    # Punctuation is removed, so "largest?" becomes "largest".
    # Common words (STOP_WORDS) are dropped.
    words = []
    current = ""

    for char in text.lower():
        if char.isalnum():
            current = current + char
        else:
            if len(current) > 0:
                words.append(current)
            current = ""

    if len(current) > 0:
        words.append(current)

    kept = []
    for word in words:
        if word not in STOP_WORDS:
            kept.append(word)

    return kept


def load_sample():
    text = read_pdf(SAMPLE_PATH)
    pieces = chunk_text(text, CHUNK_SIZE)

    chunks = []
    by_id = {}
    number = 0

    for piece in pieces:
        chunk = {}
        chunk["id"] = "sample#chunk" + str(number)
        chunk["source"] = "solar_system.pdf"
        chunk["text"] = piece["text"]
        chunk["heading"] = piece["heading"]
        chunks.append(chunk)
        by_id[chunk["id"]] = chunk
        number = number + 1

    # In-memory database. Nothing is saved to disk.
    # ChromaDB turns each text into numbers with its own small built-in model.
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(name="sample")

    batch_start = 0
    while batch_start < len(chunks):
        batch_end = batch_start + BATCH_SIZE
        batch = chunks[batch_start:batch_end]

        ids = []
        texts = []
        metadatas = []

        for chunk in batch:
            ids.append(chunk["id"])
            texts.append(chunk["text"])
            metadata = {}
            metadata["source"] = chunk["source"]
            metadata["heading"] = chunk["heading"]
            metadatas.append(metadata)

        collection.add(ids=ids, documents=texts, metadatas=metadatas)
        batch_start = batch_end

    # Keyword index (BM25) needs every chunk as a list of words.
    word_lists = []
    for chunk in chunks:
        word_lists.append(tokenize(chunk["text"]))

    store["chunks"] = chunks
    store["by_id"] = by_id
    store["collection"] = collection
    store["bm25"] = BM25Okapi(word_lists)


# ============================================================
# SEARCHING
# ============================================================

def get_top_score_indices(scores, k):
    # Finds the positions of the k highest scores, by hand:
    # pick the best, mark it as used, repeat.
    scores_copy = []
    for s in scores:
        scores_copy.append(s)

    chosen_indices = []
    count = 0

    while count < k and count < len(scores_copy):
        best_index = 0
        best_score = scores_copy[0]

        for i in range(len(scores_copy)):
            if scores_copy[i] > best_score:
                best_score = scores_copy[i]
                best_index = i

        chosen_indices.append(best_index)
        scores_copy[best_index] = -1000000000
        count = count + 1

    return chosen_indices


def search_chunks(question, top_k):
    # Returns the matching chunks WITH their full text.
    chunks = store["chunks"]
    collection = store["collection"]
    bm25 = store["bm25"]

    # found: chunk id -> list of ways it was found
    found = {}

    # Search 1: by meaning
    results = collection.query(query_texts=[question], n_results=top_k)
    meaning_ids = results["ids"][0]

    for chunk_id in meaning_ids:
        found[chunk_id] = ["meaning"]

    # Search 2: by keywords
    question_words = tokenize(question)
    bm25_scores = bm25.get_scores(question_words)
    top_indices = get_top_score_indices(bm25_scores, top_k)

    for index in top_indices:
        # A score of 0 means no word matched at all, so skip it.
        if bm25_scores[index] <= 0:
            continue

        chunk_id = chunks[index]["id"]
        if chunk_id in found:
            found[chunk_id].append("keywords")
        else:
            found[chunk_id] = ["keywords"]

    # Strongest first: found by BOTH searches, then meaning only,
    # then keywords only.
    both_list = []
    meaning_only = []
    keywords_only = []

    for chunk_id in found:
        ways = found[chunk_id]
        if len(ways) == 2:
            both_list.append(chunk_id)
        elif ways[0] == "meaning":
            meaning_only.append(chunk_id)
        else:
            keywords_only.append(chunk_id)

    ordered_ids = both_list + meaning_only + keywords_only

    output = []
    for chunk_id in ordered_ids:
        chunk = store["by_id"][chunk_id]
        item = {}
        item["source"] = chunk["source"]
        item["heading"] = chunk["heading"]
        item["found_by"] = found[chunk_id]
        item["text"] = chunk["text"]
        output.append(item)

    return output


# ============================================================
# ASKING THE LLM
# ============================================================

def build_prompt(question, chunks):
    context = ""
    for chunk in chunks:
        context = context + "[" + chunk["source"] + " / " + chunk["heading"] + "]: " + chunk["text"] + "\n\n"

    prompt = "Answer the question using the context below as your only evidence.\n"
    prompt = prompt + "Say \"I don't have enough information.\" only if the context contains nothing related to the question.\n"
    prompt = prompt + "If the context is related but incomplete, answer with what it does say.\n"
    prompt = prompt + "Never do both: either answer from the context, or reply with only: I don't have enough information.\n"
    prompt = prompt + "Cite the source filename in your answer.\n\n"
    prompt = prompt + "Context:\n" + context + "\n"
    prompt = prompt + "Question: " + question + "\n\n"
    prompt = prompt + "Answer:"

    return prompt


def looks_like_refusal(answer):
    # Only the first 120 characters are checked, so a normal answer that
    # mentions one of these phrases later on is not treated as a refusal.
    start = answer.strip().lower()[:120]

    for phrase in REFUSAL_PHRASES:
        if phrase in start:
            return True

    return False


def call_llm(base_url, api_key, model, prompt):
    # Talks to any LLM service that uses the common "OpenAI style" format.
    # Ollama on your laptop uses it too, which is why we can test locally.
    #
    # TODO before deploying: refuse addresses that point inside the server
    # itself or a private network (like localhost or 192.168.x.x), so
    # visitors cannot make our server call things it should not.
    url = base_url.rstrip("/") + "/chat/completions"

    headers = {}
    headers["Authorization"] = "Bearer " + api_key
    headers["Content-Type"] = "application/json"

    message = {}
    message["role"] = "user"
    message["content"] = prompt

    body = {}
    body["model"] = model
    body["messages"] = [message]
    body["temperature"] = 0
    body["max_tokens"] = 300

    # try / except means: "try this. If it fails with an error, run the
    # except part instead of crashing."
    try:
        response = httpx.post(url, headers=headers, json=body, timeout=180)
    except Exception as error:
        # HTTPException stops here and sends an error message back to
        # whoever called our endpoint.
        raise HTTPException(
            status_code=502,
            detail="Could not reach the LLM at " + url + ". " + str(error)
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail="The LLM service returned an error (code "
                   + str(response.status_code) + "): " + response.text[:300]
        )

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    usage = data.get("usage", {})

    result = {}
    result["answer"] = answer
    result["prompt_tokens"] = usage.get("prompt_tokens", 0)
    result["answer_tokens"] = usage.get("completion_tokens", 0)
    return result


# ============================================================
# GET FULL PDF TEXT
# ============================================================

def get_pdf_text():
    # Returns the full extracted text from solar_system.pdf
    text = read_pdf(SAMPLE_PATH)
    return text


# ============================================================
# LOAD THE SAMPLE WHEN THE SERVER STARTS
# ============================================================

start_time = time.time()
load_sample()
print("Sample loaded: " + str(len(store["chunks"])) + " chunks in "
      + str(round(time.time() - start_time, 1)) + " seconds")

# Cache the PDF text so we don't re-read it every time
store["pdf_text"] = get_pdf_text()


# ============================================================
# THE ENDPOINTS
# ============================================================

# A class with type hints describes the data a visitor must send.
# FastAPI checks it automatically and builds the form on /docs from it.
class AskRequest(BaseModel):
    question: str
    base_url: str
    api_key: str
    model: str


@app.get("/")
def root():
    # Serve index.html from current directory
    return FileResponse("index.html", media_type="text/html")


@app.get("/health")
def health():
    result = {}
    result["status"] = "ok"
    result["chunks_loaded"] = len(store["chunks"])
    return result


@app.get("/pdf")
def get_pdf():
    # Returns the full extracted text from solar_system.pdf
    result = {}
    result["text"] = store["pdf_text"]
    return result


# "question: str" and "top_k: int = 3" are type hints. FastAPI reads them
# to know that question is required text and top_k is an optional number
# (default 3). That is also how it builds the input boxes on /docs.
@app.get("/search")
def search(question: str, top_k: int = 3):
    results = search_chunks(question, top_k)

    # Show only the start of each chunk, to keep the page readable.
    previews = []
    for item in results:
        preview = {}
        preview["source"] = item["source"]
        preview["heading"] = item["heading"]
        preview["found_by"] = item["found_by"]
        preview["text"] = item["text"][:400]
        previews.append(preview)

    response = {}
    response["question"] = question
    response["results"] = previews
    return response


# POST, not GET, because the request carries a key. A key in a web
# address (GET) could end up in logs. In the body of a POST it does not.
# The key is used for this one request and is never stored.
@app.post("/ask")
def ask(request: AskRequest):
    chunks_found = search_chunks(request.question, 4)
    chunks_for_llm = chunks_found[:MAX_CHUNKS_TO_LLM]

    prompt = build_prompt(request.question, chunks_for_llm)
    llm_result = call_llm(request.base_url, request.api_key, request.model, prompt)

    answer = llm_result["answer"]

    # Show sources only when the model really answered.
    sources = []
    if not looks_like_refusal(answer):
        for chunk in chunks_for_llm:
            already_listed = False
            for source in sources:
                if source["source"] == chunk["source"] and source["heading"] == chunk["heading"]:
                    already_listed = True
            if not already_listed:
                entry = {}
                entry["source"] = chunk["source"]
                entry["heading"] = chunk["heading"]
                sources.append(entry)

    tokens = {}
    tokens["prompt"] = llm_result["prompt_tokens"]
    tokens["answer"] = llm_result["answer_tokens"]

    response = {}
    response["answer"] = answer
    response["sources"] = sources
    response["tokens"] = tokens
    return response
