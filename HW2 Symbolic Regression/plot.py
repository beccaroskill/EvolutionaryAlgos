import pandas as pd
import matplotlib.pyplot as plt

# PLOT DOT PLOT
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.set_title('Dot Plot of Genetic Program')
ax.set_ylabel('Fitness (RMS)')
ax.set_xlabel('Generationi')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

dots_df = pd.read_csv("results/dotplot/dots.csv")
s = [0.5 for n in range(len(dots_df.Index[:10000]))]
ax.scatter(dots_df.Index[:10000], dots_df.Distance[:10000], s=s, color="turquoise")

plt.savefig('dot-plot.png'.format('results/dotplot'), dpi=200)
plt.show()
