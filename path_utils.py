import os
import re
from pathlib import Path

def convert_windows_path_to_wsl(windows_path: str) -> str:
    r"""
    Windows形式のパスをWSL形式のパスに変換する
    
    Examples:
        C:\Users\user\Pictures -> /mnt/c/Users/user/Pictures
        D:\Photos\image.jpg -> /mnt/d/Photos/image.jpg
        \\server\share\folder -> \\server\share\folder (UNCパスはそのまま)
    """
    if not windows_path:
        return windows_path
    
    # 既にLinux/WSLパスの場合はそのまま返す
    if windows_path.startswith('/'):
        return windows_path
    
    # UNCパス（\\server\share）の場合はそのまま返す
    if windows_path.startswith('\\\\'):
        return windows_path
    
    # Windows形式のパス（C:\Users\...）を変換
    if re.match(r'^[A-Za-z]:[\\\/]', windows_path):
        # ドライブレター（C:）を取得
        drive_letter = windows_path[0].lower()
        # パスの残りの部分を取得
        path_remainder = windows_path[3:]  # C:\ の部分を除く
        
        # バックスラッシュをスラッシュに変換
        path_remainder = path_remainder.replace('\\', '/')
        
        # WSL形式のパスを構築
        wsl_path = f"/mnt/{drive_letter}/{path_remainder}"
        
        return wsl_path
    
    # その他の場合はそのまま返す
    return windows_path

def convert_wsl_path_to_windows(wsl_path: str) -> str:
    r"""
    WSL形式のパスをWindows形式のパスに変換する
    
    Examples:
        /mnt/c/Users/user/Pictures -> C:\Users\user\Pictures
        /mnt/d/Photos/image.jpg -> D:\Photos\image.jpg
        \\server\share\folder -> \\server\share\folder (UNCパスはそのまま)
    """
    if not wsl_path:
        return wsl_path
    
    # UNCパスの場合はそのまま返す
    if wsl_path.startswith('\\\\'):
        return wsl_path
    
    # Windows形式のパスの場合はそのまま返す
    if re.match(r'^[A-Za-z]:[\\\/]', wsl_path):
        return wsl_path.replace('/', '\\')
    
    # WSL形式のパス（/mnt/c/...）を変換
    if wsl_path.startswith('/mnt/'):
        # ドライブレターを取得
        drive_letter = wsl_path[5].upper()
        # パスの残りの部分を取得
        path_remainder = wsl_path[7:]  # /mnt/c/ の部分を除く
        
        # スラッシュをバックスラッシュに変換
        path_remainder = path_remainder.replace('/', '\\')
        
        # Windows形式のパスを構築
        windows_path = f"{drive_letter}:\\{path_remainder}"
        
        return windows_path
    
    # その他の場合はそのまま返す
    return wsl_path

def normalize_path(path: str) -> str:
    """
    パスを正規化する（Windows -> WSL変換 + パスの正規化）
    """
    # Windows -> WSL変換
    converted_path = convert_windows_path_to_wsl(path)
    
    # パス区切り文字を統一し、余分なスラッシュを削除
    normalized = os.path.normpath(converted_path)
    
    # Windowsのパス区切り文字をLinuxのものに変換
    normalized = normalized.replace('\\', '/')
    
    return normalized

def validate_path_exists(path: str) -> bool:
    """
    パスが存在するかチェック（Windows -> WSL変換後）
    """
    converted_path = normalize_path(path)
    return os.path.exists(converted_path)