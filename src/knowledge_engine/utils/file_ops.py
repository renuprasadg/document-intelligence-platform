"""
File operation utilities
"""
import json
from pathlib import Path
from typing import Any, Dict

from knowledge_engine.core.exceptions import ValidationError


def read_text_file(filepath: str) -> str:
    """Read text file"""
    path = Path(filepath)
    if not path.exists():
        raise ValidationError(f"File not found: {filepath}")
    if not path.is_file():
        raise ValidationError(f"Not a file: {filepath}")
    try:
        return path.read_text(encoding='utf-8')
    except Exception as e:
        raise ValidationError(f"Error reading file {filepath}: {e}")


def write_text_file(filepath: str, content: str) -> None:
    """Write text file"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding='utf-8')
    except Exception as e:
        raise ValidationError(f"Error writing file {filepath}: {e}")


def read_json_file(filepath: str) -> Dict[str, Any]:
    """Read JSON file"""
    content = read_text_file(filepath)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in {filepath}: {e}")


def write_json_file(filepath: str, data: Dict[str, Any], indent: int = 2) -> None:
    """Write JSON file"""
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=False)
        write_text_file(filepath, content)
    except (TypeError, ValueError) as e:
        raise ValidationError(f"Error serializing JSON: {e}")


def ensure_directory(dirpath: str) -> None:
    """Ensure directory exists"""
    Path(dirpath).mkdir(parents=True, exist_ok=True)
