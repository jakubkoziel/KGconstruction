import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

all_results = []

for f in os.listdir(r"D:\masters_fine-tune\eval-checkpoints"):

    if 'checkpoint-' in f:
        with open(os.path.join(r"D:\masters_fine-tune\eval-checkpoints", f, 'eval_results.json'), 'r',
                  encoding='utf-8') as file:
            results = json.load(file) | {'step': int(f.split('-')[1])}
        all_results.append(results)

df = pd.DataFrame(all_results)
df.sort_values(by='step', inplace=True)
df[['eval_accuracy', 'eval_f1', 'eval_loss', 'eval_precision', 'eval_recall', 'step']].to_csv('tmp_remove.csv')
print(df)

### PLOT

# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot main metrics on primary axis with different markers
main_metrics = ['eval_f1', 'eval_precision', 'eval_recall']
colors = ['#ff7f0e', '#2ca02c', '#9467bd']  # Orange, Green, Purple
markers = ['o', '^', 'D']  # Circle, Triangle, Diamond

# Store all plot objects for legend
plots = []

for metric, color, marker in zip(main_metrics, colors, markers):
    line, = ax1.plot(df['step'], df[metric],
                     color=color,
                     linewidth=2,
                     marker=marker,
                     markersize=7,
                     markevery=3)
    plots.append((line, metric.replace('eval_', '').title()))

# Create secondary axis for loss
ax2 = ax1.twinx()
loss_color = '#d62728'  # Red
loss_line, = ax2.plot(df['step'], df['eval_loss'],
                      color=loss_color,
                      linewidth=2,
                      linestyle='--',
                      marker='s',
                      markersize=7,
                      markevery=3)
plots.append((loss_line, 'Loss'))

# Configure axes
ax1.set_xlabel('Step', fontsize=14)
ax1.set_ylabel('Validation score (on dev split)', fontsize=14)
ax1.set_ylim(0.84, .91)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.tick_params(axis='both', labelsize=12)

ax2.set_ylabel('Loss', fontsize=14, color=loss_color)
ax2.tick_params(axis='y', labelcolor=loss_color, labelsize=12)
ax2.set_ylim(0.095, .13)

# Create unified legend
legend_elements = [Line2D([0], [0],
                          color=plot[0].get_color(),
                          marker=plot[0].get_marker(),
                          linestyle=plot[0].get_linestyle(),
                          label=plot[1]) for plot in plots]

ax1.legend(handles=legend_elements,
           loc='lower right',
           fontsize=14)

#plt.title('Babelscape/wikineural-multilingual-ner fine-tuning on ReDocRED', fontsize=16, pad=20)
plt.xticks(df['step'][::2])
fig.tight_layout()
plt.savefig('training_metrics_babel.png', dpi=300, bbox_inches='tight')
plt.savefig('training_metrics_babel.pdf', format='pdf', bbox_inches='tight')
