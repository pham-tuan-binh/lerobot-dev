from urllib import response
import requests
from typing import List, Optional
import threading
import time

class LatestOnlySender:
    def __init__(self, session, url):
        self.session = session
        self.url = url
        self._latest_data = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def send(self, data):
        with self._lock:
            self._latest_data = data  # overwrite previous data

    def _run(self):
        while self._running:
            data = None
            with self._lock:
                if self._latest_data is not None:
                    data = self._latest_data
                    self._latest_data = None  # clear so we don't send duplicates

            if data:
                try:
                    response = self.session.post(
                        f"{self.url}/led",
                        data=data,
                        headers={'Content-Type': 'text/plain'},
                        timeout=5
                    )
                    response.raise_for_status()
                except Exception as e:
                    print(f"POST failed: {e}")
            else:
                time.sleep(0.01)  # sleep briefly to avoid busy wait

class LeLampHead:
    def __init__(self, server_url: str = "http://10.8.101.201", led_count: int = 39):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        self.led_count = led_count
        self._intensity = 0
        self.sender = LatestOnlySender(self.session, self.server_url)
        self.sender.start()

    def read_leds(self) -> Optional[List[str]]:
        """Read all LED colors from server
        
        Returns:
            List of hex color strings (e.g., ["#ff0000", "#00ff00", ...])
        """
        try:
            response = self.session.get(f"{self.server_url}/led", timeout=5)
            response.raise_for_status()
            
            # Arduino returns comma-separated hex colors
            colors_str = response.text.strip()
            if colors_str:
                return colors_str.split(',')
            return []
        except requests.RequestException as e:
            print(f"Error reading LEDs: {e}")
            return None
        
    def is_connected(self) -> bool:
        """Check if the head server is reachable"""
        try:
            # response = self.session.get(f"{self.server_url}/led", timeout=5)
            # return response.status_code == 200
            return True
        except requests.RequestException:
            return False
    
    def write_leds(self, colors: List[str]) -> bool:
        """Write LED colors to server
        
        Args:
            colors: List of hex color strings (e.g., ["#ff0000", "#00ff00"])
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure we have exactly led_count colors
            colors_to_send = colors[:self.led_count]
            
            # Pad with black if not enough colors provided
            while len(colors_to_send) < self.led_count:
                colors_to_send.append("#000000")
            
            # Arduino expects comma-separated hex colors as plain text
            data = ','.join(colors_to_send)

            # Use LatestOnlySender to avoid sending duplicates and fire and forget
            self.sender.send(data)
            
            return True
        except requests.RequestException as e:
            print(f"Error writing LEDs: {e}")
            return False
    
    def set_led(self, led_id: int, r: int, g: int, b: int) -> bool:
        """Set individual LED color
        
        Args:
            led_id: LED index (0-9 for 10 LEDs)
            r, g, b: RGB color values (0-255)
            
        Returns:
            True if successful, False otherwise
        """
        if led_id < 0 or led_id >= self.led_count:
            print(f"LED ID {led_id} out of range (0-{self.led_count-1})")
            return False
            
        # Read current colors
        current_colors = self.read_leds()
        if current_colors is None:
            # Initialize with all black if read fails
            current_colors = ["#000000"] * self.led_count
        
        # Update specific LED
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        current_colors[led_id] = hex_color
        
        return self.write_leds(current_colors)
    
    @property
    def intensity(self):
        return self._intensity
    
    @intensity.setter
    def intensity(self, value: int) -> bool:
        """Set the intensity of all LEDs

        Args:
            intensity: Intensity value (0 to 255)

        Returns:
            True if successful, False otherwise
        """

        # Constraint intensity to 0-255 range
        if value < 0:
            value = 0
        elif value > 255:
            value = 255

        self._intensity = value

        # Update all LEDs to the same intensity
        colors = [f"#{value:02x}{value:02x}{value:02x}"] * self.led_count
        return self.write_leds(colors)

    def turn_off(self) -> bool:
        """Turn off all LEDs"""
        return self.set_all_leds(0, 0, 0)