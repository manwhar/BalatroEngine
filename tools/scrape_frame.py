import subprocess
import sys
from pathlib import Path

try:
    import yt_dlp
except ImportError:
    print("yt-dlp is not installed. Please install it using: pip install yt-dlp")
    sys.exit(1)


# Edit these values before running this script.
VIDEO_URL = "https://www.youtube.com/watch?v=_ZuzKjGVRPs"
DOWNLOAD_VIDEO = True
VIDEO_FILENAME = r"assets\sample_inputs\videos\bu_magic_deck_11-13-26"
DOWNLOAD_FRAME = False
FRAME_TIMESTAMP = "00:01:23.500"
FRAME_FILENAME = "scraped_frame.jpg"

SCRIPT_DIR = Path(__file__).parent
FRAME_OUTPUT_DIR = SCRIPT_DIR.parent / "assets" / "sample_inputs" / "scraped_frames"
VIDEO_OUTPUT_DIR = SCRIPT_DIR.parent / "assets" / "sample_inputs" / "videos"


def get_1080p_stream(video_url: str) -> str:
    """Extracts the direct stream URL for the 1080p video format."""
    # Force 1080p height. yt-dlp will throw an error if it doesn't exist.
    ydl_opts = {
        "quiet": True,
        "format": "bestvideo[height=1080]/best[height=1080]",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(video_url, download=False)
        except yt_dlp.utils.DownloadError as e:
            if "Requested format is not available" in str(e):
                raise ValueError("1080p format is not available for this video.") from e
            raise ValueError(f"Failed to fetch video info: {e}") from e

        if "requested_formats" in info:
            for f in info["requested_formats"]:
                if f.get("vcodec") != "none":
                    return f["url"]
        elif "url" in info:
            return info["url"]

        raise ValueError("Could not find a valid stream URL in the extracted info.")


def download_video(video_url: str, output_path: Path) -> None:
    """Download the complete video and merge separate audio/video streams."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "outtmpl": str(output_path),
    }

    print(f"Downloading video to: {output_path}")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
    except yt_dlp.utils.DownloadError as e:
        raise ValueError(f"Failed to download video: {e}") from e


def grab_frame(video_url: str, timestamp: str, output_path: Path) -> None:
    """Grabs a specific frame using the stream URL and ffmpeg."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching 1080p stream URL for: {video_url}")

    try:
        stream_url = get_1080p_stream(video_url)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Extracting frame at {timestamp}...")

    # We use ffmpeg to quickly seek to the timestamp and dump 1 frame.
    # OpenCV's cv2.VideoCapture seeking is notoriously broken on remote HLS/DASH streams.
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite if exists
        "-ss",
        timestamp,  # Fast seek (must be before -i)
        "-i",
        stream_url,  # Input stream URL
        "-vframes",
        "1",  # Only grab 1 frame
        "-q:v",
        "2",  # High quality JPEG
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Successfully saved frame to: {output_path}")
    except FileNotFoundError:
        print("Error: 'ffmpeg' is not installed or not in your PATH.", file=sys.stderr)
        print("ffmpeg is required to extract frames from the stream.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ffmpeg failed: {e.stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    if not VIDEO_URL:
        raise ValueError("Set VIDEO_URL before running scrape_frame.py")

    if DOWNLOAD_VIDEO:
        download_video(VIDEO_URL, VIDEO_OUTPUT_DIR / VIDEO_FILENAME)

    if DOWNLOAD_FRAME:
        grab_frame(
            VIDEO_URL,
            FRAME_TIMESTAMP,
            FRAME_OUTPUT_DIR / FRAME_FILENAME,
        )
