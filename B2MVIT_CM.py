import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cm = np.array([[277, 2],
               [0, 359]])

row_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
row_percent = np.round(row_percent, 2)

labels = ['Monkeypox', 'Others']

plt.figure(figsize=(6, 5))
sns.heatmap(row_percent, annot=True, fmt=".2f", cmap='Blues', xticklabels=labels, yticklabels=labels)

plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix for B2MVIT")
plt.show()
