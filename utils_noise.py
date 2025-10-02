import numpy as np

# Parameterised Salt and Pepper Noise Function
def add_saltpepper_noise(V_hat, p=0, r=0, random_state=None):
    """
    Add parameterised noise to the dataset
    
    Args:
        V_hat: Clean data matrix (pixels × images)
        p: Ratio of pixel corruption (noise level)
        r: Ratio of corrupted pixels that become white (255)
        random_state: Random seed for reproducibility
    
    Returns:
        V: Noisy data matrix
        noise_mask: Binary mask showing which pixels were corrupted
    """
    rng = np.random.RandomState(random_state)
    
    V = V_hat.copy()
    
    # Create noise mask: p of pixels will be corrupted
    total_pixels = V.size
    num_corrupt = int(p * total_pixels)
    flat_indices = rng.choice(total_pixels, size=num_corrupt, replace=False)
    noise_mask = np.zeros(V.shape, dtype=bool)
    noise_mask.flat[flat_indices] = True
    
    # r of corrupted pixels will become white
    num_white = int(r * num_corrupt)
    white_indices = rng.choice(num_corrupt, size=num_white, replace=False)
    
    # Create white/black mask for corrupted pixels
    white_pixels = np.zeros(V.shape, dtype=bool)
    white_pixels.flat[flat_indices[white_indices]] = True
    
    # Apply noise: corrupted pixels become either white (255) or black (0)
    V[noise_mask & white_pixels] = 255  # White noise
    V[noise_mask & ~white_pixels] = 0    # Black noise
    
    return V, noise_mask

def add_gaussian_noise(V_hat, sigma=0.03, noise_type='additive', random_state=None):
    rng = np.random.RandomState(random_state)
    
    V = V_hat.copy()
    
    if noise_type == 'additive':
        # Additive Gaussian noise: V = V_hat + N(0, sigma squared)
        noise_scale = sigma * 255
        noise_matrix = rng.normal(0, noise_scale, V.shape)
        V = V_hat + noise_matrix
        
    else:
        # Multiplicative Gaussian noise: V = V_hat * (1 + N(0, sigma squared))
        noise_matrix = rng.normal(0, sigma, V.shape)
        V = V_hat * (1 + noise_matrix)
        noise_matrix = V - V_hat

    # Clip values to valid range
    V = np.clip(V, 0, 255)
    
    return V, noise_matrix

def add_block_occlusion(V_hat, image_shape, block_fraction=0.2, fill_value=0,
                        n_blocks=1, reduce=3, random_state=None):
    """
    Add square block occlusions per image column.
    image_shape: (width, height) BEFORE downscaling
    reduce: downscale factor used when building V_hat
    Returns: (V_occluded, mask) with mask=True where occluded
    """
    rng = np.random.RandomState(random_state)
    V = V_hat.copy()
    p, n = V.shape

    w0, h0 = image_shape
    w, h = w0 // reduce, h0 // reduce
    assert h * w == p, "image_shape and reduce must satisfy h*w == V_hat.shape[0]"

    block_area = max(1, int(block_fraction * (h * w)))
    side = min(max(1, int(np.floor(np.sqrt(block_area)))), h, w)

    mask = np.zeros_like(V, dtype=bool)

    for j in range(n):
        img = V[:, j].reshape(h, w)
        m2 = np.zeros((h, w), dtype=bool)
        for _ in range(n_blocks):
            r0 = rng.randint(0, h - side + 1)
            c0 = rng.randint(0, w - side + 1)
            img[r0:r0+side, c0:c0+side] = fill_value
            m2[r0:r0+side, c0:c0+side] = True
        V[:, j] = img.reshape(-1)
        mask[:, j] = m2.reshape(-1)

    V = np.clip(V, 0, 255)
    return V, mask