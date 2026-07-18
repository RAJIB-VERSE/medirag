# MediRAG v2 — Multilingual, Multi-Domain Medical RAG Chatbot 🩺🌍

A Retrieval-Augmented Generation (RAG) chatbot that answers health questions **in English, Bengali, Hindi, or any language mix**, grounded in **170K+ trusted medical Q&A entries** from 5 authoritative datasets. Cites its sources with real links and refuses to guess when it doesn't have reliable information.

> ⚠️ **Disclaimer:** This is an educational/portfolio project, not a certified medical device. It does not provide medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical concerns.

![MediRAG demo](assets/demo.gif)

---

## 🆕 What's New in v2

| Feature | v1 | v2 |
|---|---|---|
| **Language support** | English only | 🌐 Bengali + English, Hindi + English, any mix |
| **Medical coverage** | 47K entries (1 dataset) | 📚 170K+ entries (5 datasets) |
| **Refusal accuracy** | LLM-only judgment | 🎯 Similarity threshold + LLM judgment |
| **Emergency detection** | English keywords only | 🚨 English, Bengali, Hindi, transliterated |
| **Source diversity** | NIH sources only | NIH + WikiDoc + medical flashcards |

---

## 🧩 The Problem

Millions of people turn to search engines or general-purpose chatbots to self-diagnose symptoms. This leads to three failure modes:
- **Hallucinated confidence** — an AI states something medically inaccurate as fact.
- **Unreliable sources** — blogs and forums with no editorial standards rank alongside credible medical institutions.
- **Language barrier** — non-English speakers or those who mix languages (Bengali+English, Hindi+English) get no useful results from English-only medical tools.

**MediRAG v2** addresses all three by:
1. **Translating** mixed-language queries to English before retrieval (using the LLM itself)
2. **Grounding** every answer in retrieved passages from **5 trusted medical datasets**
3. **Refusing** to answer when retrieval quality is poor (using a similarity threshold), rather than hallucinating

---

## ⚙️ How It Works (Architecture)

```
User question (any language)
      │
      ▼
[Safety Check] ──► if emergency keywords detected ──► immediate safety message
      │                  (supports EN/BN/HI/transliterated)
      ▼ (no emergency)
[Normalize Query] ──► Phi-3 translates mixed-language → English
      │                  (skipped if already English)
      ▼
[Embed English query] ──► sentence-transformers (all-MiniLM-L6-v2)
      │
      ▼
[FAISS similarity search] ──► retrieves top-k passages from 170K+ docs
      │                         + similarity threshold filtering
      ▼
[Prompt construction] ──► English query + retrieved context + language instructions
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
| Embedding model | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector search | FAISS with L2 distance + similarity threshold |
| LLM | `microsoft/Phi-3-mini-4k-instruct` (4-bit via `bitsandbytes`) |
| Query translation | Phi-3 (same model, zero-shot translation) |
| Interface | Gradio |
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
4. The final cell launches a Gradio chat window with a public shareable link.

No manual dataset download needed — all datasets stream directly from Hugging Face.

> **Note:** First run takes ~10-15 minutes (embedding 170K documents). Subsequent runs are faster if the FAISS index is cached.

---

## 💬 Example Interactions

**✅ English query — grounded, cited answer**
> **Q: What is cholesterol?** *(previously refused in v1!)*
> Cholesterol is a waxy, fat-like substance found in every cell of your body... [Source: WikiDoc]

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

## 🧪 Evaluation

Tested across 20+ queries spanning five categories:

| Category | Example | Expected behavior | Result |
|---|---|---|---|
| Well-covered topic | "What is asthma?" | Grounded, cited answer | ✅ Pass |
| Previously missing topic | "What is cholesterol?" | Now answered (from WikiDoc) | ✅ Pass (was ❌ in v1) |
| Bengali + English mix | "আমার জ্বর হচ্ছে, is it dengue?" | Translate, retrieve, answer | ✅ Pass |
| Hindi transliterated | "diabetes ke lakshan" | Translate, retrieve, answer | ✅ Pass |
| Emergency (multilingual) | "সীনে মেন দর্দ" | Safety layer intercepts | ✅ Pass |
| Unrelated query | "Capital of France?" | Doesn't force medical answer | ✅ Pass |

---

## 🔍 Known Limitations

- **Translation quality** — Phi-3's zero-shot translation works well for Bengali/Hindi but may struggle with very colloquial or rare dialects.
- **Retrieval still depends on embedding similarity** — even with 170K docs, some niche topics may not have coverage.
- **Not fine-tuned** — all grounding is retrieval-based; the LLM's baseline knowledge could still leak into edge-case answers.
- **Single-turn only** — no conversation memory for follow-up questions.
- **Response language** — the model attempts to respond in the user's language, but quality varies.

---

## 🔭 Future Improvements

- Upgrade to multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) for native multilingual retrieval.
- Add conversation memory for multi-turn follow-ups.
- Deploy to Hugging Face Spaces for a permanent public demo.
- Add WHO and CDC datasets for broader international coverage.
- Fine-tune the translation step for better Bengali/Hindi accuracy.

---

## 📄 License

This project is for educational purposes. Dataset licenses: MedQuAD (CC BY 4.0), WikiDoc (CC), Medical Flashcards (CC), MedQuad-KV (CC0). Model weights follow Microsoft Phi-3 license.
