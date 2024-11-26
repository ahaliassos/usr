import os
import argparse
import subprocess
import shutil
import torchaudio
import torch


def check_ffmpeg_installed():
    """Check if ffmpeg is installed on the system."""
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is not installed or not found in PATH.")
        exit(1)


def convert_to_avi_no_audio(video_path, avi_output_path, fps=None):
    """Convert a video to AVI format without audio."""
    if fps is None:
        cmd = [
            "ffprobe",
            "-v", "0",
            "-of", "csv=p=0",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            video_path
        ]
        try:
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            r_frame_rate = result.stdout.strip()
            num, denom = r_frame_rate.split('/')
            fps = float(num) / float(denom)
        except Exception as e:
            print(f"Error obtaining video fps: {e}")
            exit(1)

    # Convert to AVI (no audio)
    command = [
        "ffmpeg",
        "-i", video_path,
        "-c:v", "libxvid",
        "-an",
        "-r", str(fps),
        avi_output_path,
        "-y"
    ]

    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Video converted to AVI without audio and saved to {avi_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting video: {e.stderr.decode().strip()}")
        exit(1)


def extract_audio_from_mp4(mp4_path, audio_output_path, sample_rate=16000):
    """Extract audio from the original MP4 and save it as WAV."""
    command = [
        "ffmpeg",
        "-i", mp4_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        audio_output_path,
        "-y"
    ]

    try:
        print("Extracting audio from MP4...")
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Audio extracted and saved to {audio_output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio: {e.stderr.decode().strip()}")
        exit(1)


def get_video_properties(avi_path):
    """Retrieve FPS and frame count from the AVI video."""
    cmd_fps = [
        "ffprobe",
        "-v", "0",
        "-of", "csv=p=0",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        avi_path
    ]
    cmd_frames = [
        "ffprobe",
        "-v", "0",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0",
        avi_path
    ]
    try:
        result_fps = subprocess.run(cmd_fps, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        r_frame_rate = result_fps.stdout.strip()
        num, denom = r_frame_rate.split('/')
        fps = float(num) / float(denom)
    except Exception as e:
        print(f"Error obtaining fps from AVI: {e}")
        exit(1)

    try:
        result_frames = subprocess.run(cmd_frames, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        frame_count_str = result_frames.stdout.strip()
        frame_count = int(frame_count_str)
    except Exception as e:
        print(f"Error obtaining frame count from AVI: {e}")
        exit(1)

    return fps, frame_count


def pad_audio(audio_output_path, expected_samples, sample_rate=16000):
    """Pad or trim audio to have exactly the expected number of samples."""
    try:
        audio, sr = torchaudio.load(audio_output_path)
    except Exception as e:
        print(f"Error loading audio for padding: {e}")
        exit(1)

    current_samples = audio.shape[1]
    print(f"Current audio samples: {current_samples}, Expected samples: {expected_samples}")

    if current_samples < expected_samples:
        padding = expected_samples - current_samples
        print(f"Padding audio with {padding} zeros.")
        audio = torch.nn.functional.pad(audio, (0, padding))
    elif current_samples > expected_samples:
        trimming = current_samples - expected_samples
        print(f"Trimming audio. Removing {trimming} samples.")
        audio = audio[:, :expected_samples]

    torchaudio.save(audio_output_path, audio, sample_rate)
    print(f"Adjusted audio saved to {audio_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert video to AVI without audio, extract audio from MP4, and add padding.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the original video file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Sample rate for audio extraction (default: 16000).")
    args = parser.parse_args()

    video_path = args.video_path
    output_dir = args.output_dir
    sample_rate = args.sample_rate

    check_ffmpeg_installed()

    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        exit(1)

    os.makedirs(output_dir, exist_ok=True)

    video_basename = os.path.basename(video_path)
    video_name, _ = os.path.splitext(video_basename)

    avi_output_path = os.path.join(output_dir, f"{video_name}.avi")
    audio_output_path = os.path.join(output_dir, f"{video_name}.wav")

    convert_to_avi_no_audio(video_path, avi_output_path)

    fps, frame_count = get_video_properties(avi_output_path)
    samples_per_frame = sample_rate // int(fps)
    expected_samples = frame_count * samples_per_frame

    extract_audio_from_mp4(video_path, audio_output_path, sample_rate=sample_rate)
    pad_audio(audio_output_path, expected_samples, sample_rate=sample_rate)

    print("Video conversion and audio extraction completed.")


if __name__ == "__main__":
    main()

"""
Usage:
    python video_audio_converter.py --video_path /path/to/video.mp4 --output_dir /path/to/output --sample_rate 16000
"""