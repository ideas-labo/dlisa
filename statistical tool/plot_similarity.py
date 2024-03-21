import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


# 启用 LaTeX 文本渲染
plt.rcParams['text.usetex'] = False
plt.rcParams['text.latex.preamble'] = r'\usepackage{mathptmx}'
plt.rcParams['text.latex.preamble'] = r'\usepackage{bm}'
# 在macOS中指定Times New Roman字体的典型路径
font_path = '/System/Library/Fonts/Supplemental/Times New Roman.ttf'
font_prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

# Data from the user's datasets

###X264
data_A_scores = [None, None, 0.18, 0.21, 0.17, 0.19, 0.19, 0.13, 0.21]
data_B_scores = [None, None, 0.56, 0.54, 0.54, 0.52, 0.56, 0.58, 0.59]

# Create the figure and the first axis
fig, ax1 = plt.subplots(figsize=(14, 6))

# Plot the data for Dataset A
ax1.plot(data_A_scores, color='#C8102E', marker='o', alpha = 1, linestyle='-', label='Dataset A')

# Label the first axis
ax1.set_ylabel('Average Similarity Score', fontsize = 26)
ax1.tick_params(axis='x', labelcolor='#C8102E')
ax1.set_xticks(range(len(data_A_scores)))

#X264
ax1.set_xticklabels(['W1', 'W7', 'W3', 'W8', 'W9', 'W2', 'W5', 'W6', 'W4'], fontproperties=font_prop, fontsize = 24)

# Create the second axis for Dataset B, sharing the same y-axis
ax2 = ax1.twiny()

# Plot the data for Dataset B
ax2.plot(data_B_scores, color='#00247D', marker='^', alpha = 1, linestyle='-', label='Dataset B')

# Label the second axis
ax2.tick_params(axis='x', labelcolor='#00247D')
ax2.set_xticks(range(len(data_B_scores)))

###X264
ax2.set_xticklabels(['W9','W3','W5', 'W2', 'W4', 'W8', 'W7', 'W1', 'W6'], fontproperties=font_prop, fontsize = 24)
# Add a horizontal baseline at y=0.6 for both axes
ax1.axhline(y=0.3, color='green', linestyle='--')

ax1.set_ylim(0.0, 0.7)
ax1.tick_params(axis='y', labelsize=20)  # 设置y轴刻度标签的大小
# Fixing legend overlapping by positioning them separately
#ax1.legend(fontproperties=font_prop)
#ax2.legend(fontproperties=font_prop)





# Show the plot
plt.show()
plt.savefig('Discussion/x264_similarity.pdf')  # 保存图形