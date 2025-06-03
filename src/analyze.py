import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

models = {
    "DeepExplan+GPT": "./result/model_data/Dgpt.csv",
    "DeepExplan+Deep": "./result/model_data/Ddeepseek.csv",
    "GptExplan+Deep": "./result/model_data/Cdeepseek.csv",
    "RAGGptExplan+Deep": "./result/model_data/CdeepseekRAG.csv",
    "GptExplan+GPT": "./result/model_data/Cgpt.csv",
    "RAGGptExplan+GPT": "./result/model_data/Cgpt_RAG.csv",
    "RAGDeepExplan+Deep": "./result/model_data/DdeepseekRAG.csv",
    "RAGDeepExplan+GPT": "./result/model_data/Dgpt_RAG.csv",
}

summary_stats = []
combined_data = []

def analyze_scores(df, source_name):
    score_series = df["score"]
    
    mean = score_series.mean()
    median = score_series.median()
    std = score_series.std()
    high_score_rate = (score_series >= 4).mean() * 100  
    mode = score_series.mode().values[0] if not score_series.mode().empty else "N/A"

    summary_stats.append({
        "Model": source_name,
        "Mean": mean,
        "Median": median,
        "Mode": mode,
        "Std Dev": std,
        "% Scores ≥ 4": high_score_rate
    })

    score_counts = score_series.value_counts().sort_index()
    for score, count in score_counts.items():
        combined_data.append({"Model": source_name, "Score": score, "Count": count})

for name, path in models.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        analyze_scores(df, name)
    else:
        print(f"[Warning] File not found: {path}")

summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv("./result/score_summary_comparison.csv", index=False)
print("\nSaved summary to: score_summary_comparison.csv")

combined_df = pd.DataFrame(combined_data)
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=combined_df, x="Score", y="Count", hue="Model")

for bars in ax.containers:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 5,                    
            f"{int(height)}",
            ha='center', va='bottom',
            fontsize=6
        )


plt.title("Score Distribution Across All Models")
plt.xlabel("Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("./result/combined_score_distribution.png")
plt.close()
print("Saved combined plot: combined_score_distribution.png")
