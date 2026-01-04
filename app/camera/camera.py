import platform
import cv2


def open_capture(src_str: str, preferred_backend: str = "auto") -> cv2.VideoCapture:
    """
    Opens camera/video source with backend selection for better compatibility.
    - src_str: "0", "1", ... for cameras, or file/stream path.
    - preferred_backend: auto|any|dshow|msmf|v4l2|avfoundation|gstreamer
    """
    source = int(src_str) if src_str.isdigit() else src_str
    system = platform.system()
    pb = preferred_backend.lower()

    backend = cv2.CAP_ANY

    if isinstance(source, int):
        # Auto choose backend based on OS (best default for webcams)
        if pb == "auto":
            if system == "Windows":
                backend = cv2.CAP_DSHOW
            elif system == "Linux":
                backend = cv2.CAP_V4L2
            else:
                backend = cv2.CAP_AVFOUNDATION
        else:
            backend_map = {
                "any": cv2.CAP_ANY,
                "dshow": cv2.CAP_DSHOW,
                "msmf": cv2.CAP_MSMF,
                "v4l2": cv2.CAP_V4L2,
                "avfoundation": cv2.CAP_AVFOUNDATION,
                "gstreamer": cv2.CAP_GSTREAMER,
            }
            if pb not in backend_map:
                raise ValueError(f"Unknown backend '{preferred_backend}'.")
            backend = backend_map[pb]

        cap = cv2.VideoCapture(source, backend)
    else:
        # File/URL streams (optional gstreamer)
        if pb == "gstreamer":
            cap = cv2.VideoCapture(source, cv2.CAP_GSTREAMER)
        else:
            cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {src_str}")

    return cap


def apply_camera_settings(cap: cv2.VideoCapture, width: int, height: int, fps: int, mjpg: bool) -> None:
    """
    Applies camera settings if supported by the device/driver.
    - MJPG can increase FPS for many USB webcams (especially on Raspberry Pi).
    """
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
