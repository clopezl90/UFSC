from scipy.signal import correlate2d
import numpy as np


def calculate_displacement(image_t0,image_t1,window,overlap):
    #step=window*overlap
    step = int(window * (1 - overlap))
    positions = []
    displacements = []

    for y in range(0,image_t0.shape[0]-window+1,step):
        for x in range(0,image_t1.shape[1]-window+1,step):
            window_t0=image_t0[y:y+window,x:x+window]
            window_t1=image_t1[y:y+window,x:x+window]

            corr = correlate2d(window_t1, window_t0, mode='same')

            max_y, max_x = np.unravel_index(np.argmax(corr), corr.shape)
            center = window // 2
            dx = max_x - center
            dy = max_y - center
            positions.append((x + center, y + center))
            displacements.append((dx, dy))
    
    print(f"step: {step}")
    print(f"image_t0 shape: {image_t0.shape}")
    print(f"window: {window}")
    print(f"positions: {positions[:10]}")
    return positions, displacements
