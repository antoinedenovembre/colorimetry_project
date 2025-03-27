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

def lab_to_xyz(lab):
    """
    Convert Lab to XYZ color space.
    """
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    
    L, a, b = lab
    fy = (L + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    
    def f_inv(t):
        delta = 6/29
        return t**3 if t > delta else 3 * (delta**2) * (t - 4/29)
    
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)
    
    return np.array([X, Y, Z])

def xyz_to_rgb(xyz):
    """
    Convert XYZ to RGB using the sRGB space matrix.
    """
    M = np.array([[ 3.2406, -1.5372, -0.4986],
              [-0.9689,  1.8758,  0.0415],
              [ 0.0557, -0.2040,  1.0570]])
    
    rgb = np.dot(M, xyz)
    
    rgb = np.clip(rgb, 0, 1) 
    return rgb

def lab_to_rgb(lab):
    """
    Convert Lab to RGB color space.
    """
    xyz = lab_to_xyz(lab)
    rgb = xyz_to_rgb(xyz)
    return rgb

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
    canvas = np.ones((height, width, 3), dtype=float) * bg_rgb

    # Center the resized image
    start_y = (height - new_h) // 2
    start_x = (width - new_w) // 2

    # Apply foreground color where the mask is True
    canvas_region = canvas[start_y:start_y+new_h, start_x:start_x+new_w, :]
    mask = img_resized > 0.5
    canvas_region[mask] = fg_rgb
    
    return canvas