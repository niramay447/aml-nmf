import numpy as np
import matplotlib.pyplot as plt

def compare_images(V_left, V_right, image_shape, num_images=1, reduce=3,
                   titles=("Original", "Noisy")):
    """
    Side-by-side visualization of two matrices' columns as images.

    Args:
        V_left:  matrix (pixels × samples) for left column (e.g., clean)
        V_right: matrix (pixels × samples) for right column (e.g., noisy/recon)
        image_shape: (width, height) of the original images BEFORE downscaling
        num_images: number of sample columns to show (from the start)
        reduce: downscale factor used when building V_left/V_right (e.g., 3)
        titles: tuple of (left_title, right_title)
    """
    p, n = V_left.shape
    assert V_right.shape[0] == p, "V_left and V_right must have same #rows (pixels)"
    assert V_right.shape[1] >= min(n, num_images), "V_right must have enough columns"

    # compute downscaled image size
    w0, h0 = image_shape
    w, h = w0 // reduce, h0 // reduce
    assert h * w == p, "image_shape and reduce must satisfy h*w == #rows of V"

    num = min(num_images, n)
    fig, axes = plt.subplots(num, 2, figsize=(8, 3 * num), squeeze=False)

    for i in range(num):
        axes[i, 0].imshow(V_left[:, i].reshape(h, w), cmap=plt.cm.gray)
        axes[i, 0].set_title(f"Image {i+1} ({titles[0]})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(V_right[:, i].reshape(h, w), cmap=plt.cm.gray)
        axes[i, 1].set_title(f"Image {i+1} ({titles[1]})")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()
