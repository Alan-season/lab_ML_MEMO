from unittest import result
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

# 设置字体为times new roman
matplotlib.rcParams['font.family'] = 'times new roman'

# 数据
classes_per_task = 10
ref = [90.40, 80.30, 78.33, 74.65, 71.74, 69.67, 68.19, 65.34, 63.10, 61.98]
cnn_top1 = [89.4, 73.8, 69.87, 68.42, 64.5, 62.62, 62.34, 56.91, 56.51, 54.24]
cnn_top5 = [99.2, 94.8, 93.13, 91.2, 89.18, 88.03, 87.04, 84.06, 83.99, 81.59]
nme_top1 = [89.8, 74.55, 73.23, 67.82, 65.76, 62.82, 61.47, 58.04, 56.48, 54.76]
nme_top5 = [99.5, 95.05, 93.97, 91.25, 90.38, 88.73, 87.53, 86.01, 84.43, 82.6]

def display_top_curves(curve_top1, curve_top5, eval_idx):
    epochs = np.multiply([i+1 for i in range(len(curve_top1))], classes_per_task)
    font_size = 16
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, curve_top1, marker='o', label=eval_idx+" Top1")
    plt.plot(epochs, curve_top5, marker='o', label=eval_idx+" Top5")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Number of classes', fontsize=font_size)
    plt.ylabel('Accuracy (%)', fontsize=font_size)
    plt.xlim([0, 102])
    plt.ylim([40, 100])
    plt.xticks(np.arange(0, 101, 25), fontsize=14)
    plt.yticks(np.arange(40, 101, 20), fontsize=14)
    plt.grid(True)
    plt.title(eval_idx+' Accuracy Curves', fontsize=font_size)
    plt.legend()
    plt.show()

def display_compare_curves(base, reprod):
    epochs = np.multiply([i+1 for i in range(len(base))], classes_per_task)
    font_size = 16
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, base, marker='o', label="Reference")
    plt.plot(epochs, reprod, marker='o', label="Reproduction")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Number of classes', fontsize=font_size)
    plt.ylabel('Accuracy (%)', fontsize=font_size)
    plt.xlim([0, 102])
    plt.ylim([40, 100])
    plt.xticks(np.arange(0, 101, 25), fontsize=14)
    plt.yticks(np.arange(40, 101, 20), fontsize=14)
    plt.grid(True)
    plt.title('Comparison', fontsize=font_size)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    display_compare_curves(ref, cnn_top1)
    # display_top_curves(cnn_top1, cnn_top5, "CNN")
    # display_top_curves(nme_top1, nme_top5, "NME")