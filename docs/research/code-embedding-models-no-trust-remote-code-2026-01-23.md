# Code Embedding Models: No trust_remote_code Required

**Research Date:** 2026-01-23
**Objective:** Find alternative code-specific embedding models that work with sentence-transformers without requiring `trust_remote_code=True`

## Executive Summary

**Best Working Code-Specific Model:** `microsoft/graphcodebert-base`

**Key Findings:**
- ✓ 2 code-specific models work without trust_remote_code
- ✗ Salesforce CodeT5+ requires trust_remote_code (not usable)
- ✓ Both Microsoft models show code-aware behavior
- GraphCodeBERT shows better code vs. natural language separation

## Tested Models

### ✓ WORKING: microsoft/graphcodebert-base

**Status:** SUCCESS ✓
**Dimensions:** 768
**Trust Remote Code Required:** No
**Code-Specific:** Yes

**Performance Metrics:**
- Python fibonacci ↔ JavaScript fibonacci similarity: 0.9344
- Python code ↔ Natural language similarity: 0.6167
- JavaScript code ↔ Natural language similarity: 0.6295

**Analysis:**
- Strong code-to-code similarity (93.44%)
- Clear separation between code and natural language (~62%)
- Best choice for code-specific embeddings without trust_remote_code
- Incorporates data flow information for better semantic understanding

**Warning:** Model shows message: "Some weights were not initialized from checkpoint and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']"
- This is expected behavior when using base models with sentence-transformers
- Model still functions correctly for embeddings

### ✓ WORKING: microsoft/codebert-base

**Status:** SUCCESS ✓
**Dimensions:** 768
**Trust Remote Code Required:** No
**Code-Specific:** Yes

**Performance Metrics:**
- Python fibonacci ↔ JavaScript fibonacci similarity: 0.9776
- Python code ↔ Natural language similarity: 0.9316
- JavaScript code ↔ Natural language similarity: 0.9022

**Analysis:**
- Very high code-to-code similarity (97.76%)
- Less separation between code and natural language (90-93%)
- May conflate code and text more than GraphCodeBERT
- Good alternative but GraphCodeBERT shows better domain separation

### ✗ FAILED: Salesforce/codet5p-110m-embedding

**Status:** FAILED ✗
**Error:** ValueError
**Reason:** "The repository contains custom code which must be executed to correctly load the model"

**Conclusion:** Cannot be used without `trust_remote_code=True`

### ✓ WORKING: sentence-transformers/all-mpnet-base-v2

**Status:** SUCCESS ✓
**Dimensions:** 768
**Trust Remote Code Required:** No
**Code-Specific:** No (general-purpose)

**Notes:**
- High-quality general-purpose embedding model
- NOT trained specifically on code
- Useful baseline but not optimal for code search

### ✓ WORKING: sentence-transformers/all-MiniLM-L6-v2

**Status:** SUCCESS ✓
**Dimensions:** 384
**Trust Remote Code Required:** No
**Code-Specific:** No (general-purpose)

**Notes:**
- Fast and efficient general-purpose model
- Smaller dimension size (384 vs 768)
- NOT code-specific

### ✓ WORKING: BAAI/bge-small-en-v1.5

**Status:** SUCCESS ✓
**Dimensions:** 384
**Trust Remote Code Required:** No
**Code-Specific:** No (general-purpose)

**Notes:**
- Popular BGE (BAAI General Embedding) model
- NOT specifically trained on code
- Good general-purpose alternative

## Recommendation

**Primary Recommendation:** `microsoft/graphcodebert-base`

**Reasoning:**
1. **Code-Specific Training:** Pre-trained on 6 programming languages (Python, Java, JavaScript, PHP, Ruby, Go)
2. **Data Flow Awareness:** Incorporates data flow representation for better semantic understanding of code logic
3. **Better Domain Separation:** Shows clearer distinction between code and natural language (62% vs 90%+ for CodeBERT)
4. **No Custom Code:** Works without trust_remote_code=True
5. **Standard Dimensions:** 768-dimensional embeddings (same as CodeBERT)

**Secondary Recommendation:** `microsoft/codebert-base`

**Use Case:** If you need very high code-to-code similarity detection (97.76% vs 93.44%)

## Implementation Example

```python
from sentence_transformers import SentenceTransformer

# Load GraphCodeBERT without trust_remote_code
model = SentenceTransformer("microsoft/graphcodebert-base", trust_remote_code=False)

# Get embedding dimensions
dims = model.get_sentence_embedding_dimension()  # Returns: 768

# Encode code
code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)
"""

embedding = model.encode(code)
print(f"Embedding shape: {embedding.shape}")  # (768,)
```

## GraphCodeBERT vs CodeBERT: Technical Comparison

### Architectural Differences

**GraphCodeBERT:**
- Considers inherent structure of code
- Incorporates data flow representation
- Pre-trained on NL-PL pairs in 6 languages
- Data flow graph appended to code during preprocessing
- Reflects program logic, not just text

**CodeBERT:**
- Text-based code representation
- No additional structural data
- Simpler architecture
- May conflate code and natural language

### Performance Benchmarks (2025)

Recent benchmarks show mixed results:

**Code Retrieval Tasks:**
- Voyage Code-3: 97.3% MRR
- GraphCodeBERT: 50.9% MRR
- CodeBERT: 11.7% MRR

**Interpretation:**
- Both models significantly underperform modern commercial models
- GraphCodeBERT outperforms CodeBERT in retrieval tasks (50.9% vs 11.7%)
- May require fine-tuning for optimal performance
- Older transformer architectures vs. modern designs

**Code Search Performance:**
- CodeCSE (zero-shot) outperforms both CodeBERT and GraphCodeBERT on all languages
- Suggests both models may need fine-tuning for production use

### Practical Limitations

**GraphCodeBERT:**
- Likely requires fine-tuning for better performance
- May not be optimal for large repository search without fine-tuning
- Better than CodeBERT but still significantly behind commercial models

**CodeBERT:**
- Performs poorly on retrieval tasks (11.7% MRR)
- Limited by text-only representation
- Requires fine-tuning for most production use cases

### Use Cases

**Both Models Are Suitable For:**
- Code search (with fine-tuning)
- Code completion
- Bug detection
- Semantic code similarity
- Cross-language code understanding

**GraphCodeBERT Advantages:**
- Better logical understanding via data flow
- More accurate code-comment alignment
- Superior code vs. natural language separation

## trust_remote_code Context (2025)

### Why Some Models Require It

Many newer embedding models on HuggingFace require `trust_remote_code=True` because they:
- Use custom architectures not in standard transformers
- Have model-specific preprocessing code
- Include specialized tokenization logic

### Security Concerns

The `trust_remote_code` parameter "should only be set to True for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine."

### Models Known to Require trust_remote_code

- Salesforce/codet5p-110m-embedding ✗
- jinaai/jina-embeddings-v4 ✗
- dunzhang/stella_en_400M_v5 ✗
- Alibaba-NLP/gte-large-en-v1.5 ✗

### Standard Models (No trust_remote_code)

- microsoft/graphcodebert-base ✓
- microsoft/codebert-base ✓
- sentence-transformers/all-MiniLM-L6-v2 ✓
- sentence-transformers/all-mpnet-base-v2 ✓
- BAAI/bge-small-en-v1.5 ✓

## Testing Methodology

### Test Setup

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model
model = SentenceTransformer(model_name, trust_remote_code=False)

# Test samples
code_sample1 = """def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)"""

code_sample2 = """function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}"""

natural_language = "This is a regular sentence about cooking pasta for dinner."

# Encode and compare
emb1 = model.encode(code_sample1)
emb2 = model.encode(code_sample2)
emb_nl = model.encode(natural_language)

# Cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Evaluation Criteria

**Code-Aware Model Indicators:**
1. High code-to-code similarity (same algorithm, different languages)
2. Lower code-to-natural-language similarity
3. Separation between domains (code vs. text)

**Results:**
- GraphCodeBERT: 93.44% code-code, 62% code-text → ✓ Code-aware
- CodeBERT: 97.76% code-code, 90%+ code-text → ✓ Code-aware (less separation)

## Migration Path

### Current Model (Hypothetical)
If currently using a model requiring `trust_remote_code=True`:

```python
# OLD (requires trust_remote_code)
model = SentenceTransformer(
    "Salesforce/codet5p-110m-embedding",
    trust_remote_code=True
)
```

### Recommended Migration

```python
# NEW (no trust_remote_code required)
model = SentenceTransformer(
    "microsoft/graphcodebert-base",
    trust_remote_code=False
)
```

### Compatibility Notes

**Dimension Changes:**
- If migrating from different dimension models, you'll need to re-index
- GraphCodeBERT: 768 dimensions
- CodeBERT: 768 dimensions
- Some alternatives: 384 dimensions (MiniLM, BGE-small)

**Performance Impact:**
- GraphCodeBERT may require fine-tuning for optimal results
- Consider performance benchmarks for your specific use case
- Test with representative code samples from your domain

## Future Considerations

### Alternative Approaches

1. **Commercial Models:** Voyage Code-3, OpenAI embeddings (97.3% MRR)
2. **Fine-tuning:** Both CodeBERT and GraphCodeBERT can be fine-tuned
3. **Hybrid Approach:** Use general-purpose models with code-specific preprocessing
4. **Newer Models:** Monitor HuggingFace for code models that don't require trust_remote_code

### Monitoring and Evaluation

- Track code search quality metrics
- A/B test GraphCodeBERT vs. CodeBERT in production
- Monitor for new code-specific models on HuggingFace
- Consider fine-tuning if base model performance is insufficient

## References and Sources

1. [GitHub - microsoft/CodeBERT: CodeBERT](https://github.com/microsoft/CodeBERT)
2. [What embedding models work best for code and technical content? - Zilliz Vector Database](https://zilliz.com/ai-faq/what-embedding-models-work-best-for-code-and-technical-content)
3. [Exploring GraphCodeBERT for Code Search: Insights and Limitations - DEV Community](https://dev.to/rebuss/exploring-graphcodebert-for-code-search-insights-and-limitations-4jm)
4. [Code Isn't Just Text: A Deep Dive into Code Embedding Models | Medium](https://medium.com/@abhilasha4042/code-isnt-just-text-a-deep-dive-into-code-embedding-models-418cf27ea576)
5. [The need to trust_remote_code attribute in SentenceTransformer model loading - GitHub Issue](https://github.com/UKPLab/sentence-transformers/issues/2272)
6. [SentenceTransformer Documentation](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html)

## Appendix: Test Results Output

### GraphCodeBERT Test Output

```
============================================================
Testing: microsoft/graphcodebert-base
============================================================
✓ Model loaded successfully
  Dimensions: 768
  Embedding shape: (768,)

Similarity Scores:
  Python fibonacci <-> JavaScript fibonacci: 0.9344
  Python code <-> Natural language: 0.6167
  JavaScript code <-> Natural language: 0.6295
  ✓ Model appears code-aware (code-code > code-text similarity)
```

### CodeBERT Test Output

```
============================================================
Testing: microsoft/codebert-base
============================================================
✓ Model loaded successfully
  Dimensions: 768
  Embedding shape: (768,)

Similarity Scores:
  Python fibonacci <-> JavaScript fibonacci: 0.9776
  Python code <-> Natural language: 0.9316
  JavaScript code <-> Natural language: 0.9022
  ✓ Model appears code-aware (code-code > code-text similarity)
```

### CodeT5+ Test Output

```
============================================================
Testing: Salesforce/codet5p-110m-embedding
============================================================
✗ Failed to load
  Error: ValueError
  Message: The repository Salesforce/codet5p-110m-embedding contains custom code
  which must be executed to correctly load the model. You can inspect the repository
  content at https://hf.co/Salesforce/codet5p-110
```

---

**Conclusion:** Use `microsoft/graphcodebert-base` for code-specific embeddings without requiring `trust_remote_code=True`. It provides the best balance of code understanding, domain separation, and security compliance.
