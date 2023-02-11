import numpy as np
import matplotlib.pyplot as plt

num_augs = 5
data_name = "imagenet"

# inv  = np.load(data_name+"_similarity_matrix_" + str(num_augs) + "augs.npy")
inv = np.load("val_imagenet1k_similarity_matrix_5_KL_augs.npy")

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
plt.savefig("unnormalized" + str(num_augs) + "_kl_augs_invariances.png")