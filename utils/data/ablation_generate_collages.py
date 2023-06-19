import cv2
import numpy as np
from perlin_noise import PerlinNoise

def generate_collages(textures, perlin_p=1, add_circle=False, rng=None):
    img_size = textures.shape
    num_class = img_size[0]
    if np.random.random() <= perlin_p:
        masks = perlin_masks(img_size, num_class, rng=rng)
    else:
        masks = polygon_masks(img_size, num_class, add_circle=add_circle, rng=rng)
    query = sum(textures[i] * masks[:, :, i:i+1] for i in range(num_class))
    return query, masks[:, :, 0]

def polygon_masks(img_size, num_class=5, add_circle=False, rng=None):
    if rng is not None:
        points = rng.integers(0, img_size[1], size=(num_class, 2))
    else:
        points = np.random.randint(0, img_size[1], size=(num_class, 2))
    xs, ys = np.meshgrid(np.arange(0, img_size[2]), np.arange(0, img_size[1]))
    dists_b = [np.sqrt((xs - p[0]) ** 2 + (ys - p[1]) ** 2) for p in points]
    voronoi = np.argmin(dists_b, axis=0)
    masks_b = np.zeros((img_size[1], img_size[2], num_class), dtype=int)
    for m in range(num_class):
        if add_circle:
            if rng is not None:
                num_c = rng.integers(1, 3)
                rs = rng.uniform(10, 50, size = (num_c,))
                cpc = rng.uniform(0, masks_b.shape[0], size = (num_c,2))
            else:
                num_c = np.random.randint(1, 3)
                rs = np.random.uniform(10, 50, size = (num_c,))
                cpc = np.random.uniform(0, masks_b.shape[0], size = (num_c,2))
            dist_center = [np.sqrt((xs - c[0]) ** 2 + (ys - c[1]) ** 2) for c in cpc]
            for i in range(len(rs)):
                mask = dist_center[i] <= rs[i]
                voronoi[mask] = m
            masks_b[voronoi == m] = 0
        masks_b[:, :, m][voronoi == m] = 1

    return masks_b

def perlin_masks(img_size, num_class=5, thres=0.1, rng=None):
    if num_class > 3:
        n = 2
    else:
        n = 3
    masks_b = np.ones((img_size[1], img_size[2], num_class), dtype=int)
    pix = int(img_size[1]/8)
    for m in range(num_class):  
        if m != 0:
            if rng is not None:
                noise = PerlinNoise(octaves=n, seed = rng.integers(0, 5000))
            else:
                noise = PerlinNoise(octaves=n)
            temp = np.asarray([[noise([i/pix, j/pix]) for j in range(pix)] for i in range(pix)])
            temp = np.where(temp>=thres, 1, 0).astype('uint8')
            temp = cv2.resize(temp, (img_size[1], img_size[2]))
            temp = cv2.GaussianBlur(temp,(7,7),cv2.BORDER_DEFAULT)
            masks_b[temp == 1] = 0
            masks_b[:,:,m] = temp

    return masks_b