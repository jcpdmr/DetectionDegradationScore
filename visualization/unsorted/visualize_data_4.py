import matplotlib.pyplot as plt
import numpy as np

# Dati per il primo istogramma (h_diff)
bins1 = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
values1 = [2410, 2059, 14769, 14749, 57375]

# Dati per il secondo istogramma (h_diff_new)
values2 = [22678, 8349, 14356, 8416, 30107]

# Primo istogramma
plt.figure(figsize=(6, 5))
plt.bar(bins1, values1, color='blue', alpha=0.7)
plt.xlabel('h score')
plt.ylabel('Number of samples')
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('h_score_distribution.png', dpi=300)
plt.close()

# Secondo istogramma
plt.figure(figsize=(6, 5))
plt.bar(bins1, values2, color='blue', alpha=0.7)
plt.xlabel('swapped h score')
plt.ylabel('Number of samples')
plt.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.savefig('swapped_h_score_distribution.png', dpi=300)
plt.close()