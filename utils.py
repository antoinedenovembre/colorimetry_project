import pandas as pd

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
