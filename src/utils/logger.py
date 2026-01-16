import logging
import sys
from typing import Optional

def setup_logger(name: str = "AFAD", level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """
    ロガーのセットアップを行う
    
    Args:
        name: ロガー名
        level: ログレベル
        log_file: ログ出力先ファイルパス（Optional）
        
    Returns:
        logging.Logger: 設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # ハンドラが既にある場合は追加しない
    if logger.handlers:
        return logger
        
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 標準出力ハンドラ
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ファイルハンドラ
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger
