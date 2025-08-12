import pytest
from pathlib import Path
import tempfile
from mcp.types import TextContent
from elevenlabs_mcp.utils import (
    ElevenLabsMcpError,
    make_error,
    is_file_writeable,
    make_output_file,
    make_output_path,
    find_similar_filenames,
    try_find_similar_files,
    handle_input_file,
    _validate_local_file,
    handle_input_file_paths,
    cleanup_temp_files,
    handle_multiple_files_output_mode,
)


def test_make_error():
    with pytest.raises(ElevenLabsMcpError):
        make_error("Test error")


def test_is_file_writeable():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assert is_file_writeable(temp_path) is True
        assert is_file_writeable(temp_path / "nonexistent.txt") is True


def test_make_output_file():
    tool = "test"
    text = "hello world"
    output_path = Path("/tmp")
    result = make_output_file(tool, text, output_path, "mp3")
    assert result.name.startswith("test_hello")
    assert result.suffix == ".mp3"


def test_make_output_path():
    with tempfile.TemporaryDirectory() as temp_dir:
        result = make_output_path(temp_dir)
        assert result == Path(temp_dir)
        assert result.exists()
        assert result.is_dir()


def test_find_similar_filenames():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_file.txt"
        similar_file = temp_path / "test_file_2.txt"
        different_file = temp_path / "different.txt"

        test_file.touch()
        similar_file.touch()
        different_file.touch()

        results = find_similar_filenames(str(test_file), temp_path)
        assert len(results) > 0
        assert any(str(similar_file) in str(r[0]) for r in results)


def test_try_find_similar_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_file.mp3"
        similar_file = temp_path / "test_file_2.mp3"
        different_file = temp_path / "different.txt"

        test_file.touch()
        similar_file.touch()
        different_file.touch()

        results = try_find_similar_files(str(test_file), temp_path)
        assert len(results) > 0
        assert any(str(similar_file) in str(r) for r in results)


def test_handle_input_file(sample_audio_file):
    """Test handle_input_file with a valid local file path."""
    file_content = b"\xff\xfb\x90\x64\x00"
    sample_audio_file.write_bytes(file_content)

    # Use .absolute() to satisfy the path validation check
    with handle_input_file(str(sample_audio_file.absolute())) as f:
        assert hasattr(f, "read")
        assert f.read() == file_content

    with pytest.raises(ElevenLabsMcpError):
        handle_input_file(str(sample_audio_file.parent / "nonexistent.mp3"))


def test_handle_input_file_data_uri():
    """Test handle_input_file with a valid data URI."""
    # base64 encoded "test"
    data_uri = "data:audio/mp3;base64,dGVzdA=="
    file_like = handle_input_file(data_uri)
    assert hasattr(file_like, "read")
    assert file_like.read() == b"test"
    assert file_like.name == "datauri.mp3"


def test_handle_input_file_non_audio_fails(temp_dir):
    """Test that a non-audio file fails with default audio check."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")
    with pytest.raises(ElevenLabsMcpError, match="is not an audio or video file"):
        # Use .absolute() to satisfy the path validation check
        handle_input_file(str(test_file.absolute()))


def test_handle_input_file_non_audio_succeeds(temp_dir):
    """Test that a non-audio file succeeds when audio check is disabled."""
    test_file = temp_dir / "test.txt"
    test_file.write_text("hello")
    # Use .absolute() to satisfy the path validation check
    with handle_input_file(str(test_file.absolute()), audio_content_check=False) as f:
        assert f.read() == b"hello"


def test_validate_local_file_is_directory(temp_dir):
    """Test that _validate_local_file fails if the path is a directory."""
    with pytest.raises(ElevenLabsMcpError, match="is not a file"):
        # Use .absolute() to satisfy the path validation check
        _validate_local_file(str(temp_dir.absolute()))


def test_handle_input_file_paths(sample_audio_file):
    """Test handle_input_file_paths with mixed local files and data URIs."""
    data_uri = "data:audio/wav;base64,dGVzdA=="
    # The path needs to be absolute for the test to pass in all environments
    # because of the check inside _validate_local_file
    paths = [str(sample_audio_file.absolute()), data_uri]

    api_paths, temp_files = handle_input_file_paths(paths)

    try:
        assert len(api_paths) == 2
        assert api_paths[0] == str(sample_audio_file.absolute())
        assert Path(api_paths[1]).exists()
        assert Path(api_paths[1]).suffix == ".wav"
        assert len(temp_files) == 1
        assert Path(temp_files[0]).exists()
    finally:
        # Ensure cleanup happens even if asserts fail
        cleanup_temp_files(temp_files)

    assert not Path(temp_files[0]).exists()  # Verify cleanup worked


def test_handle_multiple_files_output_mode_files():
    """Test multi-file handling for 'files' output mode."""
    results = [
        TextContent(type="text", text="Success. File saved as: /path/to/file1.mp3"),
        TextContent(type="text", text="Success. File saved as: /path/to/file2.mp3"),
    ]
    output = handle_multiple_files_output_mode(
        results, "files", "Generated voice IDs are: 123, 456"
    )

    assert isinstance(output, TextContent)
    assert (
        output.text
        == "Success. Files saved at: /path/to/file1.mp3, /path/to/file2.mp3. Generated voice IDs are: 123, 456"
    )
