"""
RAG Chatbot - Turbofan Engine Degradation
Run:
    pip install streamlit chromadb sentence-transformers groq pandas python-dotenv
    streamlit run app.py
"""

import os, pandas as pd, streamlit as st, chromadb
from dotenv import load_dotenv
from groq import Groq
from chromadb.utils import embedding_functions

load_dotenv()

# -- Settings ------------------------------------------------------------------
DATA_DIR = "./CMAPSSData"
DB_PATH  = "./chroma_db"
COLS     = ["unit","cycle","op1","op2","op3"] + [f"s{i}" for i in range(1,22)]
FAULTS   = {"FD001":"HPC Degradation","FD002":"HPC Degradation",
            "FD003":"HPC + Fan Degradation","FD004":"HPC + Fan Degradation"}

# -- Build and store chunks ----------------------------------------------------
@st.cache_resource
def load_db():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection("engines", embedding_function=ef,
                                          metadata={"hnsw:space":"cosine"})

    if col.count() == 0:
        st.info("Building ChromaDB from dataset... (approx. 2 minutes)")
        docs, ids, metas = [], [], []

        for ds in ["FD001","FD002","FD003","FD004"]:
            path = f"{DATA_DIR}/train_{ds}.txt"
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, usecols=range(26))
            df["max_cycle"] = df.groupby("unit")["cycle"].transform("max")

            for unit, g in df.groupby("unit"):
                life = int(g["max_cycle"].iloc[0])
                early = g[g["cycle"]<=10][["s3","s4","s7","s12"]].mean()
                late  = g[g["cycle"]>=g["cycle"].max()-10][["s3","s4","s7","s12"]].mean()
                text = (f"Dataset:{ds} Engine:{unit} Fault:{FAULTS[ds]} Life:{life} cycles\n"
                        f"Early-> T30:{early.s3:.1f} T50:{early.s4:.1f} Ps30:{early.s7:.2f} NRf:{early.s12:.1f}\n"
                        f"Late->  T30:{late.s3:.1f} T50:{late.s4:.1f} Ps30:{late.s7:.2f} NRf:{late.s12:.1f}\n"
                        f"T30 change:{late.s3-early.s3:+.1f}  Ps30 change:{late.s7-early.s7:+.2f}")
                docs.append(text); ids.append(f"{ds}_u{unit}")
                metas.append({"dataset":ds,"unit":int(unit),"life":life,"fault":FAULTS[ds]})

        for ds in ["FD001","FD002","FD003","FD004"]:
            path = f"{DATA_DIR}/train_{ds}.txt"
            if not os.path.exists(path): continue
            df = pd.read_csv(path, sep=r"\s+", header=None, names=COLS, usecols=range(26))
            lives = df.groupby("unit")["cycle"].max()
            text = (f"Summary {ds}: {FAULTS[ds]}\n"
                    f"Engines:{len(lives)}  Min:{lives.min()}  Max:{lives.max()}  Mean:{lives.mean():.0f} cycles")
            docs.append(text); ids.append(f"{ds}_summary")
            metas.append({"dataset":ds,"unit":-1,"life":int(lives.mean()),"fault":FAULTS[ds]})

        for i in range(0, len(docs), 50):
            col.upsert(ids=ids[i:i+50], documents=docs[i:i+50], metadatas=metas[i:i+50])
        st.success(f"ChromaDB built successfully - {col.count()} chunks")

    return col

# -- Streamlit UI --------------------------------------------------------------
st.set_page_config(page_title="Turbofan RAG", page_icon="✈️")
st.title("✈️ Turbofan Engine RAG Chatbot")
st.caption("CMAPSS Dataset · ChromaDB · Groq (Llama 3)")

if not os.getenv("GROQ_API_KEY"):
    st.error("Please set GROQ_API_KEY in your .env file - get it free from console.groq.com")
    st.stop()

if not os.path.exists(DATA_DIR):
    st.error(f"Dataset not found at {DATA_DIR} - please add the dataset files there")
    st.stop()

col = load_db()
client_ai = Groq(api_key=os.getenv("GROQ_API_KEY"))

if "history" not in st.session_state:
    st.session_state.history = []

# Display conversation
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("Sources: " + " · ".join(msg["sources"]))

# New question
question = st.chat_input("Ask about an engine or fault...")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving..."):
            results = col.query(query_texts=[question], n_results=5,
                                include=["documents","metadatas","distances"])
            chunks = results["documents"][0]
            src_ids = results["ids"][0]

            context = "\n\n".join(f"[{sid}]\n{txt}" for sid, txt in zip(src_ids, chunks))
            system_prompt = f"""You are an expert in the CMAPSS turbofan engine degradation dataset.
Answer ONLY based on the context below. If the information is not available, say so.
Always reply in English only.

CONTEXT:
{context}"""

            messages = ([{"role": "system", "content": system_prompt}]
                        + [{"role": m["role"], "content": m["content"]}
                           for m in st.session_state.history]
                        + [{"role": "user", "content": question}])

            try:
                resp = client_ai.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    max_tokens=800,
                    messages=messages
                )
                answer = resp.choices[0].message.content

            except Exception as e:
                answer = f"Groq API Error: {str(e)}"

        st.markdown(answer)
        st.caption("Sources: " + " · ".join(src_ids))

    st.session_state.history += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": src_ids}
    ]