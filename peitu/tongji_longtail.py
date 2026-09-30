import matplotlib.pyplot as plt
import matplotlib as mpl
import re

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

keywords = []
freqs = []

with open('data/frequency_results.txt', 'r', encoding='utf-8') as f:
    for line in f:
        match = re.match(r'Word: (.+), Frequency: (\d+)', line)
        if match:
            word = match.group(1)
            freq = int(match.group(2))
            keywords.append(word)
            freqs.append(freq)

top_n = 10
display_keywords = keywords[:top_n]
display_freqs = freqs[:top_n]

display_keywords.append("...")
display_freqs.append(max(freqs) / 10)

tail_n = 5
display_keywords.extend(keywords[-tail_n:])
display_freqs.extend(freqs[-tail_n:])

plt.figure(figsize=(14, 6))

colors = []
for i in range(len(display_keywords)):
    if i < top_n:
        colors.append("#80A1BA")
    elif display_keywords[i] == "...":
        colors.append("#888888")
    else:
        colors.append("#CCCCCC")

bars = plt.bar(range(len(display_keywords)), display_freqs, color=colors, edgecolor='black', linewidth=0.5)

for i, (kw, freq) in enumerate(zip(display_keywords, display_freqs)):
    if kw != "...":
        plt.text(i, freq + max(display_freqs) * 0.02, f"{freq:,}", 
                ha="center", va="bottom", fontsize=9, family="SimHei")

plt.xticks(range(len(display_keywords)), display_keywords, 
           fontsize=10, family="SimHei", rotation=45, ha='right')

plt.title("Long-tail Distribution of Aviation Keyword Frequencies (Top 10 + Tail 5)", 
          fontsize=14, family="Times New Roman", fontweight='bold')
plt.ylabel("Frequency", fontsize=12, family="Times New Roman")
plt.xlabel("Keywords (sorted by frequency)", fontsize=12, family="Times New Roman")
plt.yticks(fontsize=10, family="Times New Roman")

plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.xlim(-0.5, len(display_keywords) - 0.5)
plt.ylim(0, max(display_freqs) * 1.1)

plt.tight_layout()

plt.savefig("aviation_keywords_longtail.png", dpi=300, bbox_inches="tight")
plt.show()

print(f"Chart saved as: aviation_keywords_longtail.png")
print(f"Displaying: Top {top_n} high-frequency words + ellipsis + last {tail_n} low-frequency words")
print(f"Total vocabulary: {len(keywords):,}")
print(f"Top {top_n} words total frequency: {sum(freqs[:top_n]):,} ({sum(freqs[:top_n])/sum(freqs)*100:.2f}%)")
print(f"Last {tail_n} words total frequency: {sum(freqs[-tail_n:]):,} ({sum(freqs[-tail_n:])/sum(freqs)*100:.4f}%)")
