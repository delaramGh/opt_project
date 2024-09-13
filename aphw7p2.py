import matplotlib.pyplot as plt
import numpy as np
import random


def generateArray():
    table = np.random.randint(100, 999, size=(10, 10))
    for i in range(10):
        table[i][8-i] = table[0][8]
    table[9][8] = table[0][8]
    return table

def to_str(arr):
    l = []
    n = 1
    for row in arr:
        for i in row:
            l.append(f"{n}:{i}")
            n += 1
    l2 = np.array(l, dtype=np.str)
    return l2.reshape((10,10))   

def play(silent=False, pinky=0):
    arr = generateArray()
    data = to_str(arr)
    fig = plt.figure(figsize=(8,6), dpi=100)
    ax = fig.add_subplot(1,1,1)
    ax.text(0.2, 1, "Delaram Ghobari 9623083", size=20, va='center', color='pink')
    color_pink = ['lightcoral', 'pink', 'plum', 'lightpink', 'violet', 'mistyrose', 'orchid', 'palevioletred', 'thistle', 'hotpink']
    colors = ["lightcoral", "lightsalmon", "papayawhip", "khaki", "palegreen", "aquamarine", "lightcyan", "lightblue", "plum", "violet", "pink"]
    color = []
    if not pinky:
        color  = colors
    if pinky:
        color = color_pink
    cl = [[random.choice(color) for i in range(10)] for j in range(10)]
    table = ax.table(cellText=data, loc='center', cellColours=cl, fontsize=15, cellLoc='center')
    table.scale(1,2.5)
    ax.axis('off')
    if not silent:
        fig.savefig("table.png")
        plt.show()
    if silent:
        fig.savefig("table.png")
    return arr, arr[0][8]
