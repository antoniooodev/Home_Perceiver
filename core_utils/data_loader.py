import os

import cv2
import torch

REMOTE_URL = os.getenv("REMOTE_URL", "http://192.168.1.157:8000/stream")


class VideoStream:
    def __init__(self, source=0, use_ffmpeg=False):
        """
        Initialize video stream.
        Args:
            source: 0 for local webcam, or filepath/URL for video.
            use_ffmpeg: whether to enable FFmpeg backend.
        """
        if use_ffmpeg:
            # Enable FFmpeg optimizations if available
            cv2.setUseOptimized(True)
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source {source}")

    def read_frame(self, to_tensor=True):
        """
        Read one frame and optionally convert to torch.Tensor.
        Returns:
            frame_bgr (ndarray): original BGR frame
            img_tensor (FloatTensor): [1,3,H,W], values in [0,1] or None
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if to_tensor:
            img = torch.from_numpy(frame_rgb).float() / 255.0  # [H,W,3]
            img = img.permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
            return frame, img
        return frame, None

    def release(self):
        """Release capture and close any OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()


def get_video_stream(remote_url: str = None, use_ffmpeg: bool = False):
    """
    Try to open the remote_url first (o REMOTE_STREAM_URL se non esplicitato);
    on failure, fall back to local webcam (device 0).
    """
    url = remote_url or REMOTE_URL
    if url:
        try:
            return VideoStream(source=url, use_ffmpeg=use_ffmpeg)
        except RuntimeError as e:
            print(
                f"[VideoStream] Warning: cannot open '{url}', falling back to local webcam: {e}"
            )
    return VideoStream(source=0, use_ffmpeg=False)
