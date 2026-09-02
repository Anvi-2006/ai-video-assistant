import os
import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = "downloades"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


class YouTubeDownloadError(Exception):
    """User-friendly error for YouTube download failures."""
    pass


def download_youtube_audio(url: str) -> str:
    """
    Download audio from a YouTube URL.

    If YouTube blocks the request with a 403 or another yt-dlp
    download error, raise a clean user-facing error instead of
    exposing the raw yt-dlp exception.
    """

    output_path = os.path.join(
        DOWNLOAD_DIR,
        "%(title)s.%(ext)s"
    )

    ydl_opts = {
        "format": "bestaudio",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

        # Make sure the downloaded file actually exists
        if not os.path.exists(filename):
            raise YouTubeDownloadError(
                "YouTube audio could not be downloaded."
            )

        return filename

    except yt_dlp.utils.DownloadError as e:

        error_message = str(e)

        # Handle YouTube 403 specifically
        if "403" in error_message or "Forbidden" in error_message:
            raise YouTubeDownloadError(
                "YouTube blocked the audio download (HTTP 403).\n\n"
                "This is a YouTube access restriction, not an error "
                "with the AI pipeline.\n\n"
                "Please try uploading the video/audio file directly "
                "using the Local File option."
            )

        # Handle unavailable formats
        if "Requested format is not available" in error_message:
            raise YouTubeDownloadError(
                "YouTube did not provide a downloadable audio format "
                "for this video.\n\n"
                "Please try another YouTube video or upload the "
                "video/audio file directly."
            )

        # Handle other yt-dlp errors
        raise YouTubeDownloadError(
            "Unable to download this YouTube video.\n\n"
            "Please check that the video is public and accessible, "
            "or upload the video/audio file directly."
        )


def convert_to_wav(input_path: str) -> str:
    """
    Convert an audio/video file to mono 16 kHz WAV.
    """

    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    try:
        audio = AudioSegment.from_file(input_path)

        audio = (
            audio
            .set_channels(1)
            .set_frame_rate(16000)
        )

        audio.export(
            output_path,
            format="wav"
        )

        return output_path

    except Exception as e:
        raise RuntimeError(
            f"Could not convert the media file to WAV: {e}"
        )


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list:
    """
    Split WAV audio into chunks.
    """

    try:
        audio = AudioSegment.from_wav(wav_path)

        chunk_ms = chunk_minutes * 60 * 1000

        chunks = []

        for i, start in enumerate(
            range(0, len(audio), chunk_ms)
        ):
            chunk = audio[
                start:start + chunk_ms
            ]

            chunk_path = (
                f"{wav_path}_chunk_{i}.wav"
            )

            chunk.export(
                chunk_path,
                format="wav"
            )

            chunks.append(chunk_path)

        return chunks

    except Exception as e:
        raise RuntimeError(
            f"Could not process the audio file: {e}"
        )


def process_input(source: str) -> list:
    """
    Process either a YouTube URL or a local media file.
    """

    source = source.strip()

    if not source:
        raise ValueError(
            "Please provide a YouTube URL or a local file."
        )

    # ─────────────────────────────────────────────
    # YouTube URL
    # ─────────────────────────────────────────────
    if source.startswith("http://") or source.startswith("https://"):

        print("Detected YouTube URL.")
        print("Attempting to download audio...")

        downloaded_path = download_youtube_audio(source)

        print("YouTube audio downloaded.")
        print("Converting downloaded media to WAV...")

        wav_path = convert_to_wav(downloaded_path)

    # ─────────────────────────────────────────────
    # Local file
    # ─────────────────────────────────────────────
    else:

        if not os.path.exists(source):
            raise FileNotFoundError(
                f"File not found: {source}"
            )

        print("Detected local file.")
        print("Converting media to WAV...")

        wav_path = convert_to_wav(source)

    # ─────────────────────────────────────────────
    # Chunk audio
    # ─────────────────────────────────────────────

    print("Chunking audio...")

    chunks = chunk_audio(wav_path)

    if not chunks:
        raise RuntimeError(
            "No audio chunks were created."
        )

    print(
        f"Audio ready — {len(chunks)} chunk(s) created."
    )

    return chunks