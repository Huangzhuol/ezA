# Enhancing ACORD with RAG through DeepSeek-V3 and ChatGPT-4

This project evaluates how well different LLM-based pipelines retrieve and explain legal contract clauses from the ACORD dataset. It compares various combinations of explanation sources and scorers, both with and without RAG (retrieval-augmented generation).

---

## Project Overview

### Components:
- **Clause Retrieval**: Semantic search using sentence embeddings.
- **Explanation Generation**:
  - GPT-based
  - DeepSeek-based
  - With and without RAG context
- **Scoring**:
  - Relevance rated using GPT-4o-mini and DeepSeek
- **Evaluation**:
  - Aggregated scores and visualization across models

## Operation sequence

First, install the required libraries

```
pip install -r requirements.txt
```

To generate separate CSV files by running the following code, each file will produce two CSV files, corresponding to the evaluation scores of Chatgpt and Deepseek.Since each of them takes a very long time, we did not use main.py. Instead, we ran them separately. Update key with your openai and deepseek api key in your chosen file.

```
python Deepseekexplan.py
python DeepseekexplanRAG.py
python GPTexplan.py
python GPTexplanRAG.py
```

Run the analysis function to obtain the distribution graph, as well as the average value, median, mode, standard deviation, and the frequency of scores greater than 4. And the pie chart clearly represents the difference between the presence and absence of the RAG.
```
python analyze.py/analyze_pie_chart.py
```
