import os
import tempfile
import base64
import re
from pathlib import Path
from datetime import datetime
from fuzzywuzzy import fuzz
from typing import Union, Tuple
from io import BytesIO
from mcp.types import (
    EmbeddedResource,
    TextResourceContents,
    BlobResourceContents,
    TextContent,
)

# --- Constants ---
_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".flac",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
}
_AUDIO_EXTENSIONS_NO_DOT = {ext.lstrip(".") for ext in _AUDIO_EXTENSIONS}

_MIME_TO_EXT = {
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/mp4": "m4a",
    "audio/aac": "aac",
    "audio/opus": "opus",
    "text/plain": "txt",
    "application/json": "json",
    "application/xml": "xml",
    "text/html": "html",
    "text/csv": "csv",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/epub+zip": "epub",
    "video/mp4": "mp4",
    "video/avi": "avi",
    "video/quicktime": "mov",
    "video/x-ms-wmv": "wmv",
}

_EXT_TO_MIME = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "opus": "audio/opus",
    "txt": "text/plain",
    "json": "application/json",
    "xml": "application/xml",
    "html": "text/html",
    "csv": "text/csv",
}


class ElevenLabsMcpError(Exception):
    pass


def make_error(error_text: str):
    raise ElevenLabsMcpError(error_text)


def is_data_uri(uri: str) -> bool:
    """
    Check if a string is a data URI.

    Args:
        uri: String to check

    Returns:
        bool: True if the string is a data URI, False otherwise
    """
    return uri.startswith("data:")


def parse_data_uri(data_uri: str) -> Tuple[bytes, str, str]:
    """
    Parse a data URI and extract the data, media type, and file extension.

    Args:
        data_uri: Data URI string in format: data:[<mediatype>][;base64],<data>

    Returns:
        Tuple[bytes, str, str]: (decoded_data, media_type, file_extension)

    Raises:
        ElevenLabsMcpError: If the data URI is invalid or cannot be parsed
    """
    if not is_data_uri(data_uri):
        make_error("Invalid data URI: must start with 'data:'")

    # Remove the 'data:' prefix
    uri_content = data_uri[5:]

    # Split on the first comma to separate metadata from data
    if "," not in uri_content:
        make_error("Invalid data URI: missing comma separator")

    metadata, data = uri_content.split(",", 1)

    # Parse metadata: [<mediatype>][;base64]
    is_base64 = metadata.endswith(";base64")
    if is_base64:
        media_type = metadata[:-7]  # Remove ';base64'
    else:
        media_type = metadata

    # Default media type if not specified
    if not media_type:
        media_type = "text/plain"

    # Decode the data
    try:
        if is_base64:
            decoded_data = base64.b64decode(data)
        else:
            # URL decode and encode as UTF-8
            import urllib.parse

            decoded_data = urllib.parse.unquote(data).encode("utf-8")
    except Exception as e:
        make_error(f"Failed to decode data URI: {str(e)}")

    # Determine file extension from media type
    file_extension = get_extension_from_mime_type(media_type)

    return decoded_data, media_type, file_extension


def get_extension_from_mime_type(mime_type: str) -> str:
    """
    Get file extension from MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        str: File extension (without dot)
    """
    return _MIME_TO_EXT.get(mime_type.lower(), "bin")


def create_temp_file_from_data_uri(
    data_uri: str, audio_content_check: bool = True
) -> Path:
    """
    Create a temporary file from a data URI.

    Args:
        data_uri: Data URI string
        audio_content_check: Whether to check if the file is audio/video content

    Returns:
        Path: Path to the created temporary file

    Raises:
        ElevenLabsMcpError: If the data URI is invalid or content check fails
    """
    decoded_data, media_type, file_extension = parse_data_uri(data_uri)

    # Check if it's audio/video content if required
    if audio_content_check:
        if file_extension not in _AUDIO_EXTENSIONS_NO_DOT:
            make_error(
                f"Data URI contains non-audio/video content (detected type: {media_type})"
            )

    # Create temporary file with appropriate extension
    with tempfile.NamedTemporaryFile(
        suffix=f".{file_extension}", delete=False
    ) as temp_file:
        temp_file.write(decoded_data)
        temp_path = Path(temp_file.name)

    return temp_path


def _validate_local_file(file_path: str, audio_content_check: bool = True) -> Path:
    """
    Validate a local file path and return a Path object.

    Args:
        file_path: The path to the file.
        audio_content_check: Whether to check if the file is an audio/video file.

    Returns:
        Path: A Path object for the validated file.

    Raises:
        ElevenLabsMcpError: If the file path is invalid or the file doesn't exist.
    """
    if not os.path.isabs(file_path) and not os.environ.get("ELEVENLABS_MCP_BASE_PATH"):
        make_error(
            "File path must be an absolute path if ELEVENLABS_MCP_BASE_PATH is not set"
        )
    path = Path(file_path)
    if not path.exists() and path.parent.exists():
        parent_directory = path.parent
        similar_files = try_find_similar_files(path.name, parent_directory)
        similar_files_formatted = ",".join([str(file) for file in similar_files])
        if similar_files:
            make_error(
                f"File ({path}) does not exist. Did you mean any of these files: {similar_files_formatted}?"
            )
        make_error(f"File ({path}) does not exist")
    elif not path.exists():
        make_error(f"File ({path}) does not exist")
    elif not path.is_file():
        make_error(f"File ({path}) is not a file")

    if audio_content_check and not check_audio_file(path):
        make_error(f"File ({path}) is not an audio or video file")

    return path


def handle_input_file_paths(
    file_paths: list[str], audio_content_check: bool = True
) -> Tuple[list[str], list[Path]]:
    """
    Handle a list of input file paths, creating temp files for data URIs.

    Args:
        file_paths: List of file paths or data URIs
        audio_content_check: Whether to check if files are audio/video content

    Returns:
        tuple: (list of file paths for API, list of temp files to cleanup)
    """
    api_file_paths = []
    temp_files_to_cleanup = []

    for file_path in file_paths:
        if is_data_uri(file_path):
            # Create temp file for data URI
            temp_path = create_temp_file_from_data_uri(file_path, audio_content_check)
            api_file_paths.append(str(temp_path.absolute()))
            temp_files_to_cleanup.append(temp_path)
        else:
            # For regular files, we need to validate the path but return the original path string
            # since the API expects file paths, not file handles
            path = _validate_local_file(file_path, audio_content_check)
            api_file_paths.append(str(path.absolute()))

    return api_file_paths, temp_files_to_cleanup


def cleanup_temp_files(temp_files: list[Path]):
    """
    Clean up temporary files, ignoring any errors.

    Args:
        temp_files: List of temporary file paths to clean up
    """
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass  # Ignore cleanup errors


def create_file_like_from_data_uri(
    data_uri: str, audio_content_check: bool = True
) -> BytesIO:
    """
    Create a file-like BytesIO object from a data URI.

    Args:
        data_uri: Data URI string
        audio_content_check: Whether to check if the file is audio/video content

    Returns:
        BytesIO: File-like object containing the decoded data

    Raises:
        ElevenLabsMcpError: If the data URI is invalid or content check fails
    """
    decoded_data, media_type, file_extension = parse_data_uri(data_uri)

    # Check if it's audio/video content if required
    if audio_content_check:
        if file_extension not in _AUDIO_EXTENSIONS_NO_DOT:
            make_error(
                f"Data URI contains non-audio/video content (detected type: {media_type})"
            )

    # Create BytesIO object that acts like a file
    bio = BytesIO(decoded_data)
    bio.name = f"datauri.{file_extension}"  # Some APIs might need a name
    return bio


def is_file_writeable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent_dir = path.parent
    return os.access(parent_dir, os.W_OK)


def make_output_file(
    tool: str, text: str, output_path: Path, extension: str, full_id: bool = False
) -> Path:
    id = text if full_id else text[:5]

    output_file_name = f"{tool}_{id.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
    return output_path / output_file_name


def make_output_path(
    output_directory: str | None, base_path: str | None = None
) -> Path:
    output_path = None
    if output_directory is None:
        output_path = Path.home() / "Desktop"
    elif not os.path.isabs(output_directory) and base_path:
        output_path = Path(os.path.expanduser(base_path)) / Path(output_directory)
    else:
        output_path = Path(os.path.expanduser(output_directory))
    if not is_file_writeable(output_path):
        make_error(f"Directory ({output_path}) is not writeable")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def find_similar_filenames(
    target_file: str, directory: Path, threshold: int = 70
) -> list[tuple[str, int]]:
    """
    Find files with names similar to the target file using fuzzy matching.

    Args:
        target_file (str): The reference filename to compare against
        directory (str): Directory to search in (defaults to current directory)
        threshold (int): Similarity threshold (0 to 100, where 100 is identical)

    Returns:
        list: List of similar filenames with their similarity scores
    """
    target_filename = os.path.basename(target_file)
    similar_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if (
                filename == target_filename
                and os.path.join(root, filename) == target_file
            ):
                continue
            similarity = fuzz.token_sort_ratio(target_filename, filename)

            if similarity >= threshold:
                file_path = Path(root) / filename
                similar_files.append((file_path, similarity))

    similar_files.sort(key=lambda x: x[1], reverse=True)

    return similar_files


def try_find_similar_files(
    filename: str, directory: Path, take_n: int = 5
) -> list[Path]:
    similar_files = find_similar_filenames(filename, directory)
    if not similar_files:
        return []

    filtered_files = []

    for path, _ in similar_files[:take_n]:
        if check_audio_file(path):
            filtered_files.append(path)

    return filtered_files


def check_audio_file(path: Path) -> bool:
    return path.suffix.lower() in _AUDIO_EXTENSIONS


def handle_input_file(
    file_path: str, audio_content_check: bool = True
) -> Union[BytesIO, object]:
    """
    Handle input file, always returning an open file handle.

    Args:
        file_path: File path or data URI
        audio_content_check: Whether to check if the file is audio/video content

    Returns:
        Union[BytesIO, file]: Open file handle (BytesIO for data URIs, open file for paths)
    """
    # Check if input is a data URI
    if is_data_uri(file_path):
        return create_file_like_from_data_uri(file_path, audio_content_check)

    # Original file path handling logic
    path = _validate_local_file(file_path, audio_content_check)

    # Return an open file handle
    file_handle = open(path, "rb")
    return file_handle


def handle_large_text(
    text: str, max_length: int = 10000, content_type: str = "content"
):
    """
    Handle large text content by saving to temporary file if it exceeds max_length.

    Args:
        text: The text content to handle
        max_length: Maximum character length before saving to temp file
        content_type: Description of the content type for user messages

    Returns:
        str: Either the original text or a message with temp file path
    """
    if len(text) > max_length:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file.write(text)
            temp_path = temp_file.name

        return f"{content_type.capitalize()} saved to temporary file: {temp_path}\nUse the Read tool to access the full {content_type}."

    return text


def parse_conversation_transcript(transcript_entries, max_length: int = 50000):
    """
    Parse conversation transcript entries into a formatted string.
    If transcript is too long, save to temporary file and return file path.

    Args:
        transcript_entries: List of transcript entries from conversation response
        max_length: Maximum character length before saving to temp file

    Returns:
        tuple: (transcript_text_or_path, is_temp_file)
    """
    transcript_lines = []
    for entry in transcript_entries:
        speaker = getattr(entry, "role", "Unknown")
        text = getattr(entry, "message", getattr(entry, "text", ""))
        timestamp = getattr(entry, "timestamp", None)

        if timestamp:
            transcript_lines.append(f"[{timestamp}] {speaker}: {text}")
        else:
            transcript_lines.append(f"{speaker}: {text}")

    transcript = (
        "\n".join(transcript_lines) if transcript_lines else "No transcript available"
    )

    # Check if transcript is too long for LLM context window
    if len(transcript) > max_length:
        # Create temporary file
        temp_file = tempfile.SpooledTemporaryFile(
            mode="w+", max_size=0, encoding="utf-8"
        )
        temp_file.write(transcript)
        temp_file.seek(0)

        # Get a persistent temporary file path
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as persistent_temp:
            persistent_temp.write(transcript)
            temp_path = persistent_temp.name

        return (
            f"Transcript saved to temporary file: {temp_path}\nUse the Read tool to access the full transcript.",
            True,
        )

    return transcript, False


def get_mime_type(file_extension: str) -> str:
    """
    Get MIME type for a given file extension.

    Args:
        file_extension: File extension (with or without dot)

    Returns:
        str: MIME type string
    """
    # Remove leading dot if present
    ext = file_extension.lstrip(".")

    return _EXT_TO_MIME.get(ext.lower(), "application/octet-stream")


def generate_resource_uri(filename: str) -> str:
    """
    Generate a resource URI for a given filename.

    Args:
        filename: The filename to generate URI for

    Returns:
        str: Resource URI in format elevenlabs://filename
    """
    return f"elevenlabs://{filename}"


def create_resource_response(
    file_data: bytes, filename: str, file_extension: str
) -> EmbeddedResource:
    """
    Create a proper MCP EmbeddedResource response.

    Args:
        file_data: Raw file data as bytes
        filename: Name of the file
        file_extension: File extension for MIME type detection

    Returns:
        EmbeddedResource: Proper MCP resource object
    """
    mime_type = get_mime_type(file_extension)
    resource_uri = generate_resource_uri(filename)

    # For text files, use TextResourceContents
    if mime_type.startswith("text/"):
        try:
            text_content = file_data.decode("utf-8")
            return EmbeddedResource(
                type="resource",
                resource=TextResourceContents(
                    uri=resource_uri, mimeType=mime_type, text=text_content
                ),
            )
        except UnicodeDecodeError:
            # Fall back to binary if decode fails
            pass

    # For binary files (audio, etc.), use BlobResourceContents
    base64_data = base64.b64encode(file_data).decode("utf-8")
    return EmbeddedResource(
        type="resource",
        resource=BlobResourceContents(
            uri=resource_uri, mimeType=mime_type, blob=base64_data
        ),
    )


def handle_output_mode(
    file_data: bytes,
    output_path: Path,
    filename: str,
    output_mode: str,
    success_message: str = None,
) -> Union[TextContent, EmbeddedResource]:
    """
    Handle different output modes for file generation.

    Args:
        file_data: Raw file data as bytes
        output_path: Path where file should be saved
        filename: Name of the file
        output_mode: Output mode ('files', 'resources', or 'both')
        success_message: Custom success message for files mode (optional)

    Returns:
        Union[TextContent, EmbeddedResource]: TextContent for 'files' mode,
                                            EmbeddedResource for 'resources' and 'both' modes
    """
    file_extension = Path(filename).suffix.lstrip(".")
    full_file_path = output_path / filename

    if output_mode == "files":
        # Save to disk and return TextContent with success message
        output_path.mkdir(parents=True, exist_ok=True)
        with open(full_file_path, "wb") as f:
            f.write(file_data)

        if success_message and "{file_path}" in success_message:
            message = success_message.replace("{file_path}", str(full_file_path))
        else:
            message = success_message or f"Success. File saved as: {full_file_path}"
        return TextContent(type="text", text=message)

    elif output_mode == "resources":
        # Return as EmbeddedResource without saving to disk
        return create_resource_response(file_data, filename, file_extension)

    elif output_mode == "both":
        # Save to disk AND return as EmbeddedResource
        output_path.mkdir(parents=True, exist_ok=True)
        with open(full_file_path, "wb") as f:
            f.write(file_data)
        return create_resource_response(file_data, filename, file_extension)

    else:
        raise ValueError(
            f"Invalid output mode: {output_mode}. Must be 'files', 'resources', or 'both'"
        )


def handle_multiple_files_output_mode(
    results: list[Union[TextContent, EmbeddedResource]],
    output_mode: str,
    additional_info: str = None,
) -> Union[TextContent, list[EmbeddedResource]]:
    """
    Handle different output modes for multiple file generation.

    Args:
        results: List of results from handle_output_mode calls
        output_mode: Output mode ('files', 'resources', or 'both')
        additional_info: Additional information to include in files mode message

    Returns:
        Union[TextContent, list[EmbeddedResource]]: TextContent for 'files' mode,
                                                   list of EmbeddedResource for 'resources' and 'both' modes
    """
    if output_mode == "files":
        # Extract file paths from TextContent objects and create combined message
        file_paths = []
        for result in results:
            if isinstance(result, TextContent):
                # Extract file path from the success message
                text = result.text
                if "File saved as: " in text:
                    # This parsing is simple and assumes the path is the last part of the message.
                    # It works for the default success message from handle_output_mode.
                    path = text.split("File saved as: ")[1].strip()
                    file_paths.append(path)

        message = f"Success. Files saved at: {', '.join(file_paths)}"
        if additional_info:
            message += f". {additional_info}"

        return TextContent(type="text", text=message)

    elif output_mode in ["resources", "both"]:
        # Return list of EmbeddedResource objects
        embedded_resources = []
        for result in results:
            if isinstance(result, EmbeddedResource):
                embedded_resources.append(result)

        if not embedded_resources:
            return TextContent(type="text", text="No files generated")

        return embedded_resources

    else:
        raise ValueError(
            f"Invalid output mode: {output_mode}. Must be 'files', 'resources', or 'both'"
        )


def get_output_mode_description(output_mode: str) -> str:
    """
    Generate a dynamic description for the current output mode.

    Args:
        output_mode: The current output mode ('files', 'resources', or 'both')

    Returns:
        str: Description of how the tool will behave based on the output mode
    """
    if output_mode == "files":
        return "Saves output file to directory (default: $HOME/Desktop)"
    elif output_mode == "resources":
        return "Returns output as base64-encoded MCP resource"
    elif output_mode == "both":
        return "Saves file to directory (default: $HOME/Desktop) AND returns as base64-encoded MCP resource"
    else:
        return "Output behavior depends on ELEVENLABS_MCP_OUTPUT_MODE setting"
