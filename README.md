# MediRAG v3 — Multilingual, Hybrid Search, Caching & Memory 🩺🌍

A Retrieval-Augmented Generation (RAG) chatbot that answers health questions **in English, Bengali, Hindi, or any language mix**, grounded in **170K+ trusted medical Q&A entries** from 5 authoritative datasets. Cites its sources with real links and refuses to guess when it doesn't have reliable information.

> ⚠️ **Disclaimer:** This is an educational/portfolio project, not a certified medical device. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

---

## 🆕 What's New in v3

| Feature | v2 | v3 |
|---|---|---|
| **Startup Speed** | ~15 minutes (re-embeds every run) | ⚡ **< 30 seconds** (Caches indices to Google Drive) |
| **Retrieval Engine** | Vector Search only (FAISS) | 🔀 **Hybrid Search** (FAISS semantic + BM25 keyword matching via RRF) |
| **Embeddings** | English-only (`all-MiniLM-L6-v2`) | 🌐 **Native Multilingual** (`paraphrase-multilingual-MiniLM-L12-v2`) |
| **Conversation** | Single-turn only | 🧠 **Multi-turn Memory** (remembers last 3 exchanges for context) |

---

## 🧩 The Problem

Millions of people turn to search engines or general-purpose chatbots to self-diagnose symptoms. This leads to three failure modes:
- **Hallucinated confidence** — an AI states something medically inaccurate as fact.
- **Unreliable sources** — blogs and forums with no editorial standards rank alongside credible medical institutions.
- **Language barrier** — non-English speakers or those who mix languages (Bengali+English, Hindi+English) get no useful results from English-only medical tools.

**MediRAG v3** addresses all three by:
1. **Multilingual Embedding & Translation**: Using `paraphrase-multilingual-MiniLM` to natively understand foreign queries, and Phi-3 to translate mixed-language concepts.
2. **Grounding** every answer in retrieved passages from **5 trusted medical datasets**.
3. **Hybrid Search**: Fusing semantic meaning (FAISS) and exact medical terminology matching (BM25) to ensure perfect retrieval.
4. **Refusing** to answer when retrieval quality is poor (using a similarity threshold), rather than hallucinating.

---

## ⚙️ How It Works (Architecture)

```
User question (any language) + Memory (Last 3 turns)
      │
      ▼
[Safety Check] ──► if emergency keywords detected ──► immediate safety message
      │                  (supports EN/BN/HI/transliterated)
      ▼ (no emergency)
[Hybrid Retrieval]
      │
      ├─────► FAISS Vector Search (Multilingual Embeddings)
      │
      └─────► BM25 Keyword Search (English Translated Query)
      │
      ▼
[Reciprocal Rank Fusion] ──► Merges FAISS and BM25 results, outputs Top 5
      │
      ▼
[Prompt construction] ──► User query + Top 5 Docs + Chat History + Language instructions
      │
      ▼
[LLM generation] ──► Phi-3-mini-4k-instruct (4-bit quantized)
      │                  responds in user's original language
      ▼
Answer + cited sources
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Embedding model | `sentence-transformers` (`paraphrase-multilingual-MiniLM-L12-v2`) |
| Vector search | FAISS with L2 distance |
| Keyword search | `rank_bm25` (Okapi BM25) |
| Re-ranking | Reciprocal Rank Fusion (RRF) |
| LLM | `microsoft/Phi-3-mini-4k-instruct` (4-bit via `bitsandbytes`) |
| Storage | Google Drive (Caching FAISS/BM25 indices) |
| Interface | Gradio (`ChatInterface`) |
| Compute | Google Colab (T4 GPU) |

### 📚 Data Sources (170K+ Q&A pairs)

| Dataset | Size | Coverage | License |
|---|---|---|---|
| [MedQuAD](https://huggingface.co/datasets/lavita/MedQuAD) | ~47K | NIH diseases/conditions (12 NIH websites) | CC BY 4.0 |
| [WikiDoc](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc) | ~67K | General medical knowledge | CC |
| [WikiDoc Patient Info](https://huggingface.co/datasets/medalpaca/medical_meadow_wikidoc_patient_information) | ~5K | Patient-friendly medical info | CC |
| [Medical Flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards) | ~33K | Pharmacology, terminology, clinical | CC |
| [MedQuad-KV](https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset) | ~16K | Curated medical Q&A | CC0 |

---

## 🚀 Running It

1. Open `notebooks/MediRAG_Colab.ipynb` in Google Colab.
2. Set the runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`.
3. Run all cells top to bottom.
4. **Important**: Colab will ask you for permission to mount your Google Drive. This is required to save the search indices so you don't have to wait 15 minutes every time you restart!
5. The final cell launches a Gradio chat window with a public shareable link.

> **Note:** First run takes ~10-15 minutes to embed 170K documents. Subsequent runs will load from Google Drive in **< 30 seconds**.

---

## 💬 Example Interactions

**✅ Multi-turn Memory**
> **User:** What is asthma?
> **MediRAG:** Asthma is a condition in which your airways narrow... [Source: WikiDoc]
> **User:** What are its triggers? *(Model remembers context)*
> **MediRAG:** Asthma triggers include airborne allergens, respiratory infections, physical activity... [Source: MedQuAD]

**🌐 Bengali + English mix — translated and answered**
> **Q: আমার মাথা ব্যথা হচ্ছে, what should I do?**
> 🌐 Translated: "I have a headache, what should I do?"
> Headaches can have many causes including tension, dehydration, stress...

**🌐 Hindi (transliterated) — understood and answered**
> **Q: diabetes ke lakshan kya hain**
> 🌐 Translated: "What are the symptoms of diabetes?"
> The symptoms of diabetes include being very thirsty, frequent urination...

**⚠️ Borderline topic — honest refusal**
> **Q: What is the best cryptocurrency to invest in?**
> I'm sorry, but this is not a medical question. I can only help with health-related queries.

**🚨 Emergency — multilingual safety layer**
> **Q: বুকে ব্যথা হচ্ছে** *(Bengali: "Having chest pain")*
> ⚠️ This sounds like it could be a medical emergency. Please call **112** (India) or **911** (USA)...

---

## 📄 License

This project is for educational purposes. Dataset licenses: MedQuAD (CC BY 4.0), WikiDoc (CC), Medical Flashcards (CC), MedQuad-KV (CC0). Model weights follow Microsoft Phi-3 license.
