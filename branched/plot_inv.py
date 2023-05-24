import numpy as np
import matplotlib.pyplot as plt

num_augs = 5
data_name = "imagenet1k"

# inv  = np.load(data_name+"_similarity_matrix_" + str(num_augs) + "augs.npy")
# inv = np.load("../val_imagenet1k_similarity_matrix_5_KL_adapters_augs.npy")
inv = np.load("../moco/similarity.npy")
# inv = np.array([0.3729, 0.9151, 0.4714, 0.3308, 0.9781], 
#                [0.8228, 0.9497, 0.8767, 0.8881, 0.9891],
#                [0.7876, 0.9357, 0.3748, 0.5791, 0.9798], 
#                [0.8091, 0.9417, 0.7])
print(inv)
x_tick_names = ['Resized Crop', 'Color Jitter', 'Grayscale', 'Gaussian Blur', 'H-Flip']
if num_augs == 5:
    x_tick_names = x_tick_names[0:5]

num_encoders = 6
x_ticks = np.arange(num_encoders)
str_x_ticks = [str(x_ticks[j] + 1) for j in range(num_encoders)]
fig, ax = plt.subplots(figsize=(48,10))
# Number of models is the index
index = np.arange(num_encoders)
bar_width = 0.1

for i in range(num_augs):
    ax.bar(index+bar_width*(i+1), inv[:, i], bar_width, label = x_tick_names[i])

# for i in range(num_augs):
#     ax.bar(index+bar_width*(i+1), inv_m[:, i], bar_width, label = x_tick_names[i], hatch='/')

ax.set_ylabel("Measured Invariances", fontsize = 18)
ax.set_xlabel("Models", fontsize = 18)
ax.axhline(0.5, linestyle='--', c = 'green')
ax.set_title("Invariances learned by the ensemble", fontsize=20)
ax.set_xticks(index+bar_width*2)
ax.set_xticklabels(str_x_ticks, fontsize=15)
plt.legend(loc='upper right')
plt.savefig("unnormalized" + str(num_augs) + "_moco_kl_augs_invariances.png")