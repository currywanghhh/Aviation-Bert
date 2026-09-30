import matplotlib.pyplot as plt
import matplotlib as mpl

# ------------------- 全局字体设置 -------------------
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['axes.unicode_minus'] = False

# ------------------- 序数辅助函数 -------------------
def ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}" + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")

# ------------------- 数据准备 -------------------
# 第一段：高频词（排名 1-6，节省空间）
kw_high  = ["Flight", "Crew", "Monitoring", "Personnel", "Alt.Landing", "Aircraft"]
rk_high  = [1, 2, 3, 4, 5, 6]
fr_high  = [23058, 22401, 19806, 13676, 10353, 10047]

# 第二段：排名 ~1000 附近的词（频次仅 7）
kw_mid   = ["Number", "Loading", "Declare", "Detour"]
rk_mid   = [998, 999, 1000, 1001]
fr_mid   = [7, 7, 7, 7]
dp_mid   = [700, 700, 700, 700]   # 放大显示高度，保证柱子可见

# 第三段：排名 ~1500 的尾部词（频次仅 1）
kw_tail  = ["Goodbye", "Language", "Detector"]
rk_tail  = [1498, 1499, 1500]
fr_tail  = [1, 1, 1]
dp_tail  = [400, 400, 400]        # 比第二段更矮，体现差异

# ------------------- 位置布局 -------------------
# 0-5: 高频段  6: 省略号1  7-10: ~1000段  11: 省略号2  12-14: 尾部段
pos_high = list(range(6))                         # 0,1,2,3,4,5
pos_mid  = [p + 7  for p in range(len(kw_mid))]  # 7,8,9,10
pos_tail = [p + 12 for p in range(len(kw_tail))] # 12,13,14

pos_ell1 = 6.0   # 省略号1 位置（高频 ↔ ~1000）
pos_ell2 = 11.0  # 省略号2 位置（~1000 ↔ 尾部）

# ------------------- 绘图 -------------------
fig, ax = plt.subplots(figsize=(13, 6))

ax.bar(pos_high, fr_high, color="#80A1BA", width=0.8)
ax.bar(pos_mid,  dp_mid,  color="#80A1BA", width=0.8)
ax.bar(pos_tail, dp_tail, color="#80A1BA", width=0.8)

# 数值标签
for pos, freq in zip(pos_high, fr_high):
    ax.text(pos, freq + 300, f"{freq:,}", ha="center", va="bottom", fontsize=9)
for pos, freq in zip(pos_mid, fr_mid):
    ax.text(pos, dp_mid[0] + 200, str(freq), ha="center", va="bottom", fontsize=9)
for pos, freq in zip(pos_tail, fr_tail):
    ax.text(pos, dp_tail[0] + 200, str(freq), ha="center", va="bottom", fontsize=9)

# 省略号文字
y_ell = max(fr_high) * 0.12
ax.text(pos_ell1, y_ell, "...", ha="center", va="center",
        fontsize=22, fontweight="bold", family="Times New Roman")
ax.text(pos_ell2, y_ell, "...", ha="center", va="center",
        fontsize=22, fontweight="bold", family="Times New Roman")

# ------------------- 横坐标：序数排名 + 词名（两行） -------------------
all_pos    = pos_high + [pos_ell1] + pos_mid + [pos_ell2] + pos_tail
all_labels = (
    [f"{ordinal(r)}\n{w}" for r, w in zip(rk_high, kw_high)]
    + ["..."]
    + [f"{ordinal(r)}\n{w}" for r, w in zip(rk_mid, kw_mid)]
    + ["..."]
    + [f"{ordinal(r)}\n{w}" for r, w in zip(rk_tail, kw_tail)]
)

ax.set_xticks(all_pos)
ax.set_xticklabels(all_labels, fontsize=8.5, family="Times New Roman")

# 单独加粗省略号刻度标签
for label in ax.get_xticklabels():
    if label.get_text() == "...":
        label.set_fontsize(16)
        label.set_fontweight("bold")

# ------------------- 图形美化 -------------------
ax.set_title("Frequency Distribution of Aviation-Specific Terms",
             fontsize=14, family="Times New Roman")
ax.set_ylabel("Frequency", fontsize=12, family="Times New Roman")
ax.tick_params(axis='y', labelsize=11)
ax.set_xlim(-0.6, pos_tail[-1] + 0.6)

plt.tight_layout()

# ------------------- 保存高分辨率图片 -------------------
plt.savefig("aviation_keywords_with_bold_xtick.png", dpi=300, bbox_inches="tight")
plt.show()
