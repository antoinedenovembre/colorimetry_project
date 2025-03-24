import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def excel_to_dataframe(file_path, sheet_name=0, **kwargs):
    """
    Convert an Excel file to a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    sheet_name : str or int, default 0
        Sheet to read. Either sheet index or name.
    **kwargs : dict
        Additional arguments to pass to pandas.read_excel
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing the Excel data
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

def get_color_from_df(df, idx):
    """
    Extracts the color at the given index from the DataFrame.
    We assume that the DataFrame contains columns 'R', 'G', and 'B'
    with values already normalized (between 0 and 1).
    """
    row = df.iloc[idx]
    r, g, b = row['R'], row['G'], row['B']
    return np.array([[[r, g, b]]], dtype=float)

def create_binarized_art_image(image_path, bg_rgb, fg_rgb, width=582, height=827):
    """
    Loads an image from 'image_path', converts it to grayscale,
    binarizes it with a threshold, then renders it on an A5 canvas (582x827 pixels).
    
    The binarized pixels (foreground) are colored with fg_rgb, and
    the background takes the color bg_rgb.

    Returns:
        canvas (np.ndarray): RGB image as a numpy array (height, width, 3)
    """
    # Load the image
    img = plt.imread(image_path)
    
    # Convert to grayscale
    if img.ndim == 3:
        if img.shape[2] > 3:
            img = img[..., :3]
        img_gray = 0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]
    else:
        img_gray = img

    # Normalize
    if img_gray.max() > 1:
        img_gray = img_gray / 255.
    
    # Binarize
    threshold = 0.5
    img_bin = (img_gray > threshold).astype(np.float32)
    
    # Resize
    img_h, img_w = img_bin.shape
    scale_factor = min(width / img_w, height / img_h) * 0.95
    new_w = int(img_w * scale_factor)
    new_h = int(img_h * scale_factor)
    
    img_pil = Image.fromarray((img_bin * 255).astype(np.uint8))
    img_resized = img_pil.resize((new_w, new_h), Image.NEAREST)
    img_resized = np.array(img_resized) / 255.
    
    # Create canvas with background color
    canvas = np.ones((height, width, 3), dtype=float) * bg_rgb[0, 0, :]

    # Center the resized image
    start_y = (height - new_h) // 2
    start_x = (width - new_w) // 2

    # Apply foreground color where the mask is True
    canvas_region = canvas[start_y:start_y+new_h, start_x:start_x+new_w, :]
    mask = img_resized > 0.5
    canvas_region[mask] = fg_rgb[0, 0, :]
    
    return canvas