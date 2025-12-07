"""
XRD Data Loader Module
Handles loading XRD data from various file formats
"""

import numpy as np
import pandas as pd
import os
import glob
from typing import Dict, List, Tuple, Optional, Union
import io


def get_file_columns(content: str) -> Tuple[List[str], pd.DataFrame]:
    """
    Get column information from file content for user selection

    Args:
        content: File content as string

    Returns:
        Tuple of (column_names, preview_dataframe)
    """
    lines = content.strip().split('\n')

    # Try to detect delimiter
    first_data_line = None
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('*'):
            first_data_line = line
            break

    if first_data_line is None:
        return [], pd.DataFrame()

    # Detect delimiter
    if '\t' in first_data_line:
        delimiter = '\t'
    elif ',' in first_data_line:
        delimiter = ','
    elif ';' in first_data_line:
        delimiter = ';'
    else:
        delimiter = None  # whitespace

    # Try to parse as dataframe
    try:
        if delimiter:
            df = pd.read_csv(io.StringIO(content), delimiter=delimiter,
                           comment='#', header=None, nrows=10)
        else:
            df = pd.read_csv(io.StringIO(content), delim_whitespace=True,
                           comment='#', header=None, nrows=10)

        # Check if first row is header
        first_row = df.iloc[0]
        is_header = any(isinstance(v, str) and not v.replace('.', '').replace('-', '').isdigit()
                       for v in first_row)

        if is_header:
            if delimiter:
                df = pd.read_csv(io.StringIO(content), delimiter=delimiter, comment='#', nrows=10)
            else:
                df = pd.read_csv(io.StringIO(content), delim_whitespace=True, comment='#', nrows=10)
            columns = list(df.columns)
        else:
            columns = [f"Column {i+1}" for i in range(len(df.columns))]
            df.columns = columns

        return columns, df

    except Exception as e:
        return [], pd.DataFrame()


def parse_with_column_selection(content: str,
                                 theta_col: int = 0,
                                 intensity_col: int = 1,
                                 delimiter: Optional[str] = None,
                                 skip_rows: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse file with user-specified column selection

    Args:
        content: File content
        theta_col: Column index for 2theta (0-based)
        intensity_col: Column index for intensity (0-based)
        delimiter: Delimiter (None for auto-detect)
        skip_rows: Number of header rows to skip

    Returns:
        Tuple of (two_theta, intensity)
    """
    lines = content.strip().split('\n')

    # Auto-detect delimiter if not specified
    if delimiter is None:
        first_data_line = None
        for line in lines[skip_rows:]:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('*'):
                first_data_line = line
                break

        if first_data_line:
            if '\t' in first_data_line:
                delimiter = '\t'
            elif ',' in first_data_line:
                delimiter = ','
            elif ';' in first_data_line:
                delimiter = ';'

    two_theta = []
    intensity = []

    for i, line in enumerate(lines):
        if i < skip_rows:
            continue

        line = line.strip()
        if not line or line.startswith('#') or line.startswith('*'):
            continue

        if delimiter:
            parts = line.split(delimiter)
        else:
            parts = line.split()

        if len(parts) > max(theta_col, intensity_col):
            try:
                two_theta.append(float(parts[theta_col]))
                intensity.append(float(parts[intensity_col]))
            except ValueError:
                continue

    return np.array(two_theta), np.array(intensity)


def detect_file_format(content: str) -> str:
    """
    Detect the format of XRD data file

    Returns:
        'xy': Simple two-column format (2theta, intensity)
        'csv': CSV with header
        'ras': Rigaku RAS format
        'raw': Bruker RAW format
        'txt': Generic text format
    """
    lines = content.strip().split('\n')

    # Check for CSV with header
    first_line = lines[0].strip()
    if ',' in first_line and not first_line.replace(',', '').replace('.', '').replace('-', '').replace(' ', '').isdigit():
        return 'csv'

    # Check for Rigaku RAS format
    if '*RAS_DATA_START' in content or '*RAS_HEADER_START' in content:
        return 'ras'

    # Check for simple xy format
    try:
        parts = lines[0].split()
        if len(parts) >= 2:
            float(parts[0])
            float(parts[1])
            return 'xy'
    except (ValueError, IndexError):
        pass

    return 'txt'


def parse_xy_format(content: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse simple two-column XRD data"""
    lines = content.strip().split('\n')
    two_theta = []
    intensity = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('*'):
            continue

        parts = line.split()
        if len(parts) >= 2:
            try:
                two_theta.append(float(parts[0]))
                intensity.append(float(parts[-1]))  # Last column as intensity
            except ValueError:
                continue

    return np.array(two_theta), np.array(intensity)


def parse_csv_format(content: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse CSV format XRD data"""
    df = pd.read_csv(io.StringIO(content))

    # Try to find 2theta and intensity columns
    theta_cols = [col for col in df.columns if '2theta' in col.lower() or 'theta' in col.lower() or 'angle' in col.lower()]
    int_cols = [col for col in df.columns if 'int' in col.lower() or 'counts' in col.lower() or 'intensity' in col.lower()]

    if theta_cols and int_cols:
        two_theta = df[theta_cols[0]].values
        intensity = df[int_cols[0]].values
    else:
        # Use first two numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 2:
            two_theta = df[numeric_cols[0]].values
            intensity = df[numeric_cols[1]].values
        else:
            raise ValueError("Cannot find 2theta and intensity columns in CSV")

    return two_theta, intensity


def parse_ras_format(content: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse Rigaku RAS format"""
    lines = content.split('\n')
    data_start = False
    two_theta = []
    intensity = []

    for line in lines:
        line = line.strip()
        if '*RAS_DATA_START' in line:
            data_start = True
            continue
        if '*RAS_DATA_END' in line:
            break
        if data_start and line:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    two_theta.append(float(parts[0]))
                    intensity.append(float(parts[1]))
                except ValueError:
                    continue

    return np.array(two_theta), np.array(intensity)


def load_xrd_file(file_or_path: Union[str, io.BytesIO, io.StringIO],
                  filename: Optional[str] = None) -> Dict:
    """
    Load a single XRD file

    Args:
        file_or_path: File path string or file-like object
        filename: Optional filename for file-like objects

    Returns:
        Dictionary with keys: 'two_theta', 'intensity', 'filename'
    """
    if isinstance(file_or_path, str):
        with open(file_or_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        filename = os.path.basename(file_or_path)
    else:
        if hasattr(file_or_path, 'read'):
            content = file_or_path.read()
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='ignore')
        else:
            content = str(file_or_path)

    # Detect format and parse
    file_format = detect_file_format(content)

    if file_format == 'xy' or file_format == 'txt':
        two_theta, intensity = parse_xy_format(content)
    elif file_format == 'csv':
        two_theta, intensity = parse_csv_format(content)
    elif file_format == 'ras':
        two_theta, intensity = parse_ras_format(content)
    else:
        two_theta, intensity = parse_xy_format(content)

    return {
        'two_theta': two_theta,
        'intensity': intensity,
        'filename': filename or 'unknown',
        'format': file_format
    }


def load_xrd_directory(directory_path: str,
                       pattern: str = '*') -> Dict[int, List]:
    """
    Load multiple XRD files from a directory

    Args:
        directory_path: Path to directory containing XRD files
        pattern: Glob pattern for file matching

    Returns:
        Dictionary with sample IDs as keys and [two_theta, intensity] as values
    """
    xrds = {}
    file_list = glob.glob(os.path.join(directory_path, pattern))

    for i, file_path in enumerate(sorted(file_list)):
        try:
            data = load_xrd_file(file_path)
            # Try to extract ID from filename
            fname = os.path.basename(file_path)
            try:
                # Try to extract number from filename
                parts = fname.replace('.', '_').split('_')
                sample_id = None
                for part in reversed(parts):
                    if part.isdigit():
                        sample_id = int(part)
                        break
                if sample_id is None:
                    sample_id = i + 1
            except:
                sample_id = i + 1

            xrds[sample_id] = [data['two_theta'].tolist(), data['intensity'].tolist()]
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue

    return xrds


def validate_xrd_data(data: Dict) -> Tuple[bool, str]:
    """
    Validate XRD data

    Returns:
        Tuple of (is_valid, error_message)
    """
    if 'two_theta' not in data or 'intensity' not in data:
        return False, "Missing required fields: two_theta or intensity"

    two_theta = np.array(data['two_theta'])
    intensity = np.array(data['intensity'])

    if len(two_theta) == 0 or len(intensity) == 0:
        return False, "Empty data arrays"

    if len(two_theta) != len(intensity):
        return False, f"Length mismatch: two_theta ({len(two_theta)}) != intensity ({len(intensity)})"

    if np.any(np.isnan(two_theta)) or np.any(np.isnan(intensity)):
        return False, "Data contains NaN values"

    if not np.all(np.diff(two_theta) >= 0):
        return False, "2theta values are not monotonically increasing"

    return True, ""


def convert_to_dict_format(files_dict: Dict) -> Dict[int, List]:
    """
    Convert loaded files to the standard dictionary format
    used by preprocessing functions

    Args:
        files_dict: Dictionary from session_state['files']

    Returns:
        Dictionary with IDs as keys and [[two_theta], [intensity]] as values
    """
    xrds = {}
    for i, (filename, data) in enumerate(files_dict.items()):
        xrds[i] = [
            data['two_theta'].tolist() if isinstance(data['two_theta'], np.ndarray) else data['two_theta'],
            data['intensity'].tolist() if isinstance(data['intensity'], np.ndarray) else data['intensity']
        ]
    return xrds


def dict_to_files_format(xrds: Dict[int, List],
                         original_files: Dict) -> Dict:
    """
    Convert back from dict format to files format

    Args:
        xrds: Dictionary with IDs as keys
        original_files: Original files dictionary to preserve metadata

    Returns:
        Updated files dictionary
    """
    result = {}
    filenames = list(original_files.keys())

    for i, (sample_id, data) in enumerate(xrds.items()):
        if i < len(filenames):
            filename = filenames[i]
            result[filename] = {
                **original_files[filename],
                'two_theta': np.array(data[0]),
                'intensity': np.array(data[1])
            }

    return result
