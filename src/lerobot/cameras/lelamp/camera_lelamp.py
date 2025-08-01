# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Provides the LelampCamera class for capturing frames from HTTP streaming cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
import requests

from lerobot.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_lelamp import ColorMode, LeLampCameraConfig

logger = logging.getLogger(__name__)


class LelampCamera(Camera):
    """
    Manages camera interactions using HTTP MJPEG streaming.

    This class provides a high-level interface to connect to, configure, and read
    frames from HTTP streaming cameras like ESP32-CAM devices. It supports both
    synchronous and asynchronous frame reading via HTTP requests.

    Example:
        ```python
        from lerobot.cameras.lelamp import LelampCamera, LelampCameraConfig

        config = LelampCameraConfig(server_url="http://10.8.7.45")
        camera = LelampCamera(config)
        camera.connect()

        # Read frames
        frame = camera.read()
        camera.disconnect()
        ```
    """

    def __init__(self, config: LeLampCameraConfig):
        super().__init__(config)

        self.config = config
        self.server_url = config.server_url.rstrip('/')
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s

        self.session: requests.Session | None = None
        self.stream_response: requests.Response | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

        self.rotation: int | None = get_cv2_rotation(config.rotation)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.server_url})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected and stream is active."""
        status = (isinstance(self.session, requests.Session) and 
                isinstance(self.stream_response, requests.Response))
        
        # Print status of each part of connection
        print(f"{self} session is_connected: {isinstance(self.session, requests.Session)}")
        print(f"{self} stream_response is_connected: {isinstance(self.stream_response, requests.Response)}")
        print(f"{self} is_connected: {status}")
        return status

    def connect(self, warmup: bool = True):
        """
        Connects to the HTTP streaming camera.

        Raises:
            DeviceAlreadyConnectedError: If already connected.
            ConnectionError: If connection fails.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        self.session = requests.Session()
        
        # Optimize session for streaming
        self.session.headers.update({
            'User-Agent': 'LelampCamera/1.0',
            'Connection': 'keep-alive',
            'Accept': 'multipart/x-mixed-replace'
        })
        
        try:
            self.stream_response = self.session.get(
                f"{self.server_url}/stream", 
                stream=True, 
                timeout=10,
                # Additional optimizations for streaming
                headers={'Cache-Control': 'no-cache'}
            )
            self.stream_response.raise_for_status()
            
            # Initialize buffer for frame parsing
            self._buffer = b''
        except requests.RequestException as e:
            if self.session:
                self.session.close()
                self.session = None
            raise ConnectionError(f"Failed to connect to {self}: {e}")

        if warmup:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                try:
                    self.read()
                    time.sleep(0.1)
                except Exception:
                    pass

        logger.info(f"{self} connected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """HTTP cameras cannot be auto-discovered."""
        return ["http://10.8.7.45"]

    def read(self, color_mode: ColorMode | None = None) -> np.ndarray:
        """
        Reads a single frame from the HTTP MJPEG stream.

        Args:
            color_mode: Override color mode for this read.

        Returns:
            np.ndarray: The captured frame.

        Raises:
            DeviceNotConnectedError: If not connected.
            RuntimeError: If frame reading fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        frame = self._read_mjpeg_frame()
        if frame is None:
            raise RuntimeError(f"{self} read failed - no frame received.")

        processed_frame = self._postprocess_image(frame, color_mode)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return processed_frame

    def _read_mjpeg_frame(self) -> np.ndarray | None:
        """Reads a single MJPEG frame from the HTTP stream with optimized parsing."""
        if not hasattr(self, '_buffer'):
            self._buffer = b''
        
        try:
            # Use larger chunk size for better performance
            for chunk in self.stream_response.iter_content(chunk_size=8192):
                self._buffer += chunk
                
                # Look for frame boundaries more efficiently
                while True:
                    # Find the start of JPEG data (skip multipart headers)
                    header_end = self._buffer.find(b'\r\n\r\n')
                    if header_end == -1:
                        break
                        
                    header = self._buffer[:header_end]
                    self._buffer = self._buffer[header_end + 4:]
                    
                    # Parse content length from header
                    header_str = header.decode('utf-8', errors='ignore')
                    if 'Content-Length:' not in header_str:
                        continue
                        
                    try:
                        # More efficient content length parsing
                        for line in header_str.split('\r\n'):
                            if line.startswith('Content-Length:'):
                                content_length = int(line.split(':', 1)[1].strip())
                                break
                        else:
                            continue
                            
                        # Ensure we have enough data for the frame
                        while len(self._buffer) < content_length:
                            chunk = next(self.stream_response.iter_content(chunk_size=8192), b'')
                            if not chunk:
                                return None
                            self._buffer += chunk
                        
                        # Extract frame data
                        frame_data = self._buffer[:content_length]
                        self._buffer = self._buffer[content_length:]
                        
                        # Skip trailing boundary
                        if self._buffer.startswith(b'\r\n'):
                            self._buffer = self._buffer[2:]
                        
                        # Decode JPEG directly without numpy conversion overhead
                        image = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        
                        if image is not None:
                            return image
                            
                    except (ValueError, StopIteration, IndexError):
                        continue
                        
        except requests.RequestException:
            return None
            
        return None

    def _postprocess_image(self, image: np.ndarray, color_mode: ColorMode | None = None) -> np.ndarray:
        """Applies color conversion and rotation to a frame."""
        requested_color_mode = self.color_mode if color_mode is None else color_mode

        if requested_color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(f"Invalid color mode '{requested_color_mode}'.")

        # Skip unnecessary processing if no changes needed
        if requested_color_mode == ColorMode.BGR and self.rotation is None:
            return image

        processed_image = image
        
        # Only convert color if needed
        if requested_color_mode == ColorMode.RGB:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

        # Only rotate if needed
        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    def _read_loop(self):
        """Background thread loop for asynchronous reading."""
        while not self.stop_event.is_set():
            try:
                color_image = self.read()

                with self.frame_lock:
                    self.latest_frame = color_image
                self.new_frame_event.set()

            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts the background read thread."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Stops the background read thread."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def async_read(self, timeout_ms: float = 2000) -> np.ndarray:
        """
        Reads the latest frame asynchronously.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            np.ndarray: The latest frame.

        Raises:
            DeviceNotConnectedError: If not connected.
            TimeoutError: If timeout reached.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            thread_alive = self.thread is not None and self.thread.is_alive()
            raise TimeoutError(f"Timed out waiting for frame from {self} after {timeout_ms} ms. Thread alive: {thread_alive}.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self):
        """
        Disconnects from the HTTP stream and cleans up resources.

        Raises:
            DeviceNotConnectedError: If already disconnected.
        """
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.stream_response is not None:
            self.stream_response.close()
            self.stream_response = None

        if self.session is not None:
            self.session.close()
            self.session = None

        logger.info(f"{self} disconnected.")


def test_camera_stream(server_url: str = "http://10.8.7.45"):
    """
    Test function to display camera stream using OpenCV.
    
    Args:
        server_url: The HTTP server URL for the camera.
    """
    print(f"Testing LelampCamera with server: {server_url}")
    print("Press 'q' to quit the stream")
    
    config = LeLampCameraConfig(server_url=server_url)
    camera = LelampCamera(config)
    
    try:
        camera.connect()
        print("Camera connected successfully!")
        
        cv2.namedWindow('Lelamp Camera Stream', cv2.WINDOW_AUTOSIZE)
        
        while True:
            try:
                frame = camera.read()
                cv2.imshow('Lelamp Camera Stream', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
            except Exception as e:
                print(f"Error reading frame: {e}")
                # Show error frame
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, f'Error: {str(e)[:50]}', (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow('Lelamp Camera Stream', error_frame)
                cv2.waitKey(1000)  # Wait 1 second before trying again
                
    except Exception as e:
        print(f"Failed to connect to camera: {e}")
        # Show connection error
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, f'Connection Error: {str(e)[:30]}', (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.namedWindow('Lelamp Camera Stream', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Lelamp Camera Stream', error_frame)
        cv2.waitKey(3000)  # Show error for 3 seconds
        
    finally:
        try:
            camera.disconnect()
        except:
            pass
        cv2.destroyAllWindows()
        print("Stream ended.")


if __name__ == "__main__":
    import sys
    
    # Allow custom server URL as command line argument
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://10.8.7.45"
    test_camera_stream(server_url)