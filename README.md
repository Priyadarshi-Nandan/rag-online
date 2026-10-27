# RAG Online Demo

A small question-answering app I built as a learning project. You ask a question about a solar system PDF. The app finds the most relevant parts of the document and sends them to a language model (LLM), which writes the answer. The LLM can be a model running on your own computer with Ollama, or a hosted service such as OpenRouter.

I built this to understand how RAG (retrieval-augmented generation) works from the inside, and to make a version other people can try in a browser. It is a student project, not a production product.

**Status:** work in progress. It runs locally. It is not deployed yet.

---

## What it does

- Loads one PDF (`docs/solar_system.pdf`) when the server starts and splits it into chunks, keeping each chunk inside one section.
- Answers questions by searching the chunks in two ways, by meaning and by keywords, and sending the best three chunks to the LLM.
- Shows the answer, the chapter names it came from, and token counts.
- Refuses to answer when the PDF does not contain the information, instead of guessing.
- Shows the full text of the PDF in the same page, so you can see what you can ask about.
- Works with any service that uses the OpenAI-style chat API. You paste in a base URL, a key and a model name.

---

## How it works

### Architecture sketch

```
  Browser (index.html)
        |
        |  question + base URL + key + model   (POST /ask)
        v
  FastAPI server (app.py)
        |
        |-- chunking.py : split PDF into sections, then into chunks (about 500 characters)
        |
        |-- ChromaDB (in memory) : finds chunks by meaning
        |-- BM25 (in memory)     : finds chunks by keywords
        |
        |  top 3 chunks + question
        v
  LLM service (Ollama, OpenRouter, or any OpenAI-style API)
        |
        v
  answer + sources + token counts  -->  back to the browser
```

### Step by step

1. When the server starts, `app.py` reads the PDF with pypdf. `chunking.py` cuts the text into chunks. A chunk never mixes two sections. Headings like `Chapter 7: Jupiter` mark where a new section begins.
2. Each chunk is stored in ChromaDB, which turns the text into numbers (called embeddings) using its small built-in model. I add the chunks four at a time, because adding them all at once used too much memory on my laptop.
3. The same chunks are also indexed with BM25, an older keyword-scoring method. Very common words such as "the" and "is" are ignored.
4. When a question comes in, both searches run. Chunks found by both searches come first, then chunks found only by meaning, then chunks found only by keywords.
5. The top three chunks go into a prompt. The prompt tells the model to answer only from that context, and to say "I don't have enough information" if the context does not cover the question.
6. The app checks the start of the answer for refusal phrases. If it finds one, it hides the sources, so the page does not list chapters that did not help.

### Project files

- `app.py`: the FastAPI server, search code, and LLM call
- `chunking.py`: the heading-aware chunker, written with basic Python and no regular expressions
- `index.html`: the single-page frontend. Settings are on the left. The chat or the PDF view is on the right.
- `docs/solar_system.pdf`: the sample document
- `requirements.txt`: the Python packages the app needs
- `.gitignore`: keeps the virtual environment, cache files, and secret files out of GitHub
- `README.md`: this file

### Endpoints

- `GET /` opens the web page
- `GET /health` checks that the server is running and shows how many chunks are loaded
- `GET /search?question=...` shows which chunks would be sent to the LLM. I use this for debugging.
- `GET /pdf` returns the full text of the PDF
- `POST /ask` answers a question. The body has four fields: `question`, `base_url`, `api_key`, and `model`.

---

## Results so far

I wrote a set of 15 questions about the solar system and used it in my main local RAG project. The Qwen 2.5 3B model answered 14 of 15 on my own test questions, and the Qwen 2.5 7B model answered 15 of 15. Those numbers come from that project, not from this demo.

For this demo, I checked a few questions by hand:
- a normal question ("Which planet is the largest?") gave the right answer with the right chapter sources
- a question the PDF cannot answer ("Mount Everest") gave the refusal, with no sources
- a wrong base URL gave a clear error message instead of a crash

Memory on my laptop stayed around 136 to 139 MB while idle. Indexing the 50 chunks took a few seconds on my laptop. I have not measured memory or indexing time on the hosting service yet.

---

## Setup (for readers)

If you want to run this on your own computer, follow these steps. I developed it on Windows with PowerShell, so the commands below use Windows syntax.

**What you need:** Python 3 (I used 3.14), Git, and either Ollama on your computer or an API key from an OpenAI-style provider.

1. Download the project:

   ```
   git clone https://github.com/Priyadarshi-Nandan/rag-online.git
   cd rag-online
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install the packages:

   ```
   python -m pip install -r requirements.txt
   ```

4. Start the server:

   ```
   python -m uvicorn app:app --reload
   ```

   The first start is slow, because ChromaDB downloads its small embedding model (about 80 MB) and the PDF gets indexed. When you see `Sample loaded: ... chunks` in the terminal, the server is ready.

5. Open `http://127.0.0.1:8000` in your browser.

6. Pick an LLM:
   - **Ollama on your computer:** install Ollama, run `ollama pull qwen2.5:3b`, then fill in Base URL `http://localhost:11434/v1`, API key `ollama`, and model `qwen2.5:3b`.
   - **OpenRouter:** fill in Base URL `https://openrouter.ai/api/v1`, your OpenRouter key, and a model name from their website.

**Important:** type only the base URL. The app adds `/chat/completions` itself. If you add it yourself, you get a 404 error.

**Note for Windows users:** if PowerShell blocks `uvicorn.exe`, always start the server with `python -m uvicorn`, as shown above.

---

## Known limitations

- Only one document is loaded: the solar system PDF. Uploads are not supported yet.
- Chat history lives only in your browser. Refreshing the page clears it.
- There is no conversation memory. Each question is answered on its own, so a follow-up like "and how big is it?" may not work.
- There is no reranker, so the ranking is sometimes imperfect. For example, the Saturn chapter sometimes ranks above the Jupiter chapter for a "largest planet" question.
- Only the top three chunks reach the LLM. An answer that needs information spread across many chunks may be incomplete.
- The server sends requests to whatever base URL the visitor enters. If I put this online without fixing it, someone could use the server to call addresses inside its own network. I have not fixed this yet. CORS is also open to all origins.
- The code is written so the API key is used for one request and not stored. I have not audited the whole project for logging, so I would not call it secure yet.
- Token counts are my own rough calculation. They are not the provider's official count.
- `requirements.txt` lists packages without version numbers, so a fresh install may get newer versions than the ones I tested.
- Everything was tested by hand. There is no automated test suite for the demo.

---

## Future plans

- Add conversation memory and rewriting of follow-up questions
- Fix the internal-address check before any public deployment
- Add a reranker to improve the order of the chunks
- Deploy the backend to Render and measure real memory and indexing time there
- Add small file uploads with a size limit, but only if the memory allows it
- Write a harder test set and compare other 7 to 8 billion parameter models
- Build a LangChain and LangGraph version of the same app, so I can compare the approaches

---

## My journey

### Starting with a basic RAG

My first version used only a vector database. The answers were often vague. When I looked closer, I saw that the chunks were cutting through headings and sentences, so the model got half a section at a time.

### Trying different chunking methods

In a separate project, I compared five chunking methods. A plain fixed-size split broke sections apart, so I wrote a heading-aware chunker. It is the same idea, written with basic Python, so I can explain every line of it.

### Adding keyword search

Meaning-based search sometimes missed exact names, such as "Olympus Mons". I added BM25 keyword search and combined the two results. The chunks found by both methods now come first.

### Making it fit in small memory

My laptop has 8 GB of RAM, and the free hosting plan gives only 512 MB. Embedding all the chunks at once pushed memory to about 496 MB. Adding them in batches of four brought the peak down to about 279 MB. That was the first time I saw how much the way you process data matters.

### Going online

I first tried an API from AI/ML API, but every request came back with "out of funds", even on a free key. I then tried OpenRouter, and it worked. Along the way I learned that the base URL must stop before `/chat/completions`, because the app adds that part.

---

If you find a bug or have a suggestion, feel free to open an issue on GitHub.
