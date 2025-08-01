import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')  # Set the backend first
import matplotlib.pyplot as plt  # Now import pyplot

def extract_grid_mask(img, closing_kernel=5, length_frac=0.8, resize_width=None, resize_height=None):
    """
    img:       BGR or grayscale input
    closing_kernel: size of small kernel to close grid dots ([3,3] works well)
    length_frac: fraction of img width/height that a line must span to be kept
    resize_width: optional width to resize image before processing
    resize_height: optional height to resize image before processing
    Returns: grid mask as np.ndarray
    """

    if resize_width is not None or resize_height is not None:
        if resize_width is None:
            resize_width = img.shape[1]
        if resize_height is None:
            resize_height = img.shape[0]
        img = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_AREA)

    # 1) Grayscale + adaptive threshold
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim==3 else img
    binarized = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=11, C=5
    )

    # 2) Close gaps (connect dots into lines)
    kernel_close = np.ones((closing_kernel, closing_kernel), np.uint8)
    closed = cv2.morphologyEx(binarized, cv2.MORPH_CLOSE, kernel_close)

    h, w = gray.shape

    # 3a) Extract horizontal lines via opening
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(w*1), 1))
    horiz_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, horiz_kernel)

    # 3b) Extract vertical lines via opening
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(h*0.5)))
    vert_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, vert_kernel)

    # 4) Filter by length: remove any connected component shorter than threshold
    def filter_by_length(mask, axis_len):
        nb_components, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        min_len = length_frac * axis_len
        for i in range(1, nb_components):
            x, y, w0, h0, area = stats[i]
            if axis_len == w:    # horizontal mask
                if w0 >= min_len:
                    out[labels == i] = 255
            else:                # vertical mask
                if h0 >= min_len:
                    out[labels == i] = 255
        return out

    horiz_clean = filter_by_length(horiz_lines, w)
    vert_clean  = filter_by_length(vert_lines,  h)

    # 5) Combine
    grid_mask = cv2.bitwise_or(horiz_clean, vert_clean)
    return grid_mask

def estimate_square_size_from_grid_mask(grid_mask):
    """
    Given a grid mask (single lead image), estimate the grid square size in pixels.
    Returns: square_size (float)
    """
    vertical_profile   = grid_mask.sum(axis=0)
    horizontal_profile = grid_mask.sum(axis=1)

    v_peaks, _ = find_peaks(vertical_profile, height=grid_mask.shape[0] * 0.5)
    h_peaks, _ = find_peaks(horizontal_profile, height=grid_mask.shape[1] * 0.5)

    v_spacings = np.diff(v_peaks)
    h_spacings = np.diff(h_peaks)

    def mode_with_tiebreak(data):
        if len(data) == 0:
            return None, 0
        vals, counts = np.unique(data, return_counts=True)
        max_count = counts.max()
        candidates = vals[counts == max_count]
        return candidates.max(), max_count

    avg_square_width,  count_w = mode_with_tiebreak(v_spacings)
    avg_square_height, count_h = mode_with_tiebreak(h_spacings)

    if count_w > count_h:
        square_size = avg_square_width
    elif count_h > count_w:
        square_size = avg_square_height
    else:
        square_size = max(avg_square_width, avg_square_height)

    if max(count_h, count_w) > 40:
        square_size *= 5

    return square_size

def plot_grid_mask(image, grid_mask, title="Detected Grid"):
    """
    Plots the original image and its detected grid mask side by side.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(grid_mask, cmap='gray')
    axs[1].set_title(title)
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()


def get_grid_square_size(cropped_lead_img, closing_kernel=5, length_frac=0.8, plot=False):
    """
    Given a cropped lead image, return the estimated grid square size in pixels.
    Optionally plot the grid mask.
    """
    grid_mask = extract_grid_mask(cropped_lead_img, closing_kernel=closing_kernel, length_frac=length_frac)
    if plot:
        plot_grid_mask(cropped_lead_img, grid_mask)
    square_size = estimate_square_size_from_grid_mask(grid_mask)
    return square_size