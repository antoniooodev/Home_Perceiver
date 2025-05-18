# data_loader.py
# Cross‐platform video capture: returns raw frame + normalized tensor

import cv2
import torch
from pathlib import Path

class VideoStream:
    def __init__(self, source=0):
        """
        Initialize video stream.
        Args:
            source: 0 for local webcam, or filepath/URL for video.
        """
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
            img = img.permute(2,0,1).unsqueeze(0)            # [1,3,H,W]
            return frame, img
        return frame, None

    def release(self):
        """Release capture and close any OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Quick test: show raw webcam feed
    vs = VideoStream(0)
    print("VideoStream opened. Press 'q' to quit.")
    while True:
        frame, _ = vs.read_frame()
        if frame is None:
            break
        cv2.imshow("Raw Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vs.release()