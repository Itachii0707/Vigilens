"""Security validation and integrity utilities for Veyraxis Sentinel."""

import hashlib
import io
import os
import re
from pathlib import Path

from PIL import Image

# Maximum allowed image pixel decompression limit (mitigates decompression bomb DoS)
SAFE_MAX_IMAGE_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = SAFE_MAX_IMAGE_PIXELS


def validate_file_size(content_length: int, max_bytes: int) -> tuple[bool, str]:
    """Validate that uploaded content does not exceed size limits."""
    if content_length <= 0:
        return False, "File is empty (0 bytes)."
    if content_length > max_bytes:
        mb = max_bytes / (1024 * 1024)
        return False, f"File exceeds maximum allowed size of {mb:.1f} MB."
    return True, ""


def validate_image_mime(
    file_bytes: bytes, allowed_types: list[str] = ["image/jpeg", "image/png", "image/webp", "image/bmp"]
) -> tuple[bool, str, str]:
    """
    Validate that the file header and contents represent a safe, legitimate image.
    Uses both magic bytes inspection and Pillow header decoding.
    """
    if not file_bytes:
        return False, "Empty file content", ""

    # Check magic byte signatures
    mime = "application/octet-stream"
    if file_bytes.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif file_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif file_bytes.startswith(b"RIFF") and len(file_bytes) > 12 and file_bytes[8:12] == b"WEBP":
        mime = "image/webp"
    elif file_bytes.startswith(b"BM"):
        mime = "image/bmp"

    if mime not in allowed_types:
        return False, f"Unsupported or dangerous file signature: {mime}. Allowed: {allowed_types}", ""

    # Deeper structural verification with Pillow to catch corrupted headers or embedded exploits
    try:
        with Image.open(io.BytesIO(file_bytes)) as img:
            img.verify()
            format_name = (img.format or "").lower()
            return True, "", f"image/{format_name}"
    except Exception as e:
        return False, f"Invalid or malformed image data: {e!s}", ""


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and injection attacks."""
    clean = os.path.basename(filename)
    clean = re.sub(r"[^\w\.\-]", "_", clean)
    clean = clean.lstrip(".")
    if not clean:
        clean = "unnamed_input"
    return clean


def compute_file_sha256(filepath: Path) -> str:
    """Compute the SHA-256 hash of a file for tamper verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_file_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Verify that a model weights file matches its expected SHA-256 checksum."""
    actual_hash = compute_file_sha256(filepath)
    return actual_hash.lower() == expected_sha256.lower()
