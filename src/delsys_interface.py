"""
delsys_interface.py
--------------------
Streams EMG from Delsys Trigno Avanti using the AeroPy SDK (pythonnet / .dll).

Two modes:
  MOCK = True   → synthetic EMG, no hardware needed (for home / demo use)
  MOCK = False  → real AeroPy stream via DelsysAPI.dll

IMPORTANT — before switching MOCK to False:
  1. Open AeroPy/TrignoBase.py and paste your Delsys key + license strings
  2. Make sure DelsysAPI.dll is in resources/ folder
  3. Trigno Base Station must be plugged in via USB

AeroPy data flow:
  ValidateBase() → ScanSensors() → SelectAllSensors()
  → Configure() → [collect GUIDs] → Start()
  → loop: CheckDataQueue() → PollData() → your buffer
  → Stop()
"""

import threading
import time
import numpy as np
from collections import deque

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MOCK          = True        # ← flip to False when Trigno base is connected
N_SENSORS     = 6           # EMG channels per thesis Table 4.41
EMG_FS        = 1925.926    # Hz — from your data.csv header (EMG 1 (1925.926))
POLL_INTERVAL = 0.020       # seconds between PollData() calls (~50 Hz poll rate)


# ─────────────────────────────────────────────
#  MOCK EMG GENERATOR
# ─────────────────────────────────────────────
class MockEMGStream:
    """
    Generates synthetic 6-channel sEMG at ~1926 Hz.
    Produces slow sinusoidal 'contractions' so the pipeline has
    something plausible to run on during demos.
    """

    def __init__(self, n_channels=N_SENSORS, fs=EMG_FS):
        self.n_channels = n_channels
        self.fs         = fs
        self._t         = 0.0
        self._lock      = threading.Lock()
        self._buffer    = deque(maxlen=int(fs * 5))
        self._running   = False

    def connect(self):
        print("[MockEMG] Mock mode — no Delsys hardware required.")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._generate, daemon=True)
        self._thread.start()
        print(f"[MockEMG] Synthetic EMG started @ {self.fs:.1f} Hz, {self.n_channels} ch")

    def stop(self):
        self._running = False

    def _generate(self):
        # Generate 25-sample bursts every ~13 ms to approximate real hardware
        batch_size = 25
        dt = 1.0 / self.fs
        sleep_per_batch = batch_size / self.fs

        while self._running:
            batch = []
            for _ in range(batch_size):
                t = self._t
                sample = np.array([
                    0.30 * np.sin(2 * np.pi * 0.5  * t) * np.random.randn(),  # FCR
                    0.20 * np.sin(2 * np.pi * 0.4  * t) * np.random.randn(),  # ECR
                    0.25 * np.sin(2 * np.pi * 0.3  * t) * np.random.randn(),  # FCU
                    0.40 * np.sin(2 * np.pi * 0.6  * t) * np.random.randn(),  # FDS
                    0.35 * np.sin(2 * np.pi * 0.45 * t) * np.random.randn(),  # EDC
                    0.20 * np.sin(2 * np.pi * 0.55 * t) * np.random.randn(),  # PT
                ], dtype=np.float32)
                batch.append(sample)
                self._t += dt
            with self._lock:
                self._buffer.extend(batch)
            time.sleep(sleep_per_batch)

    def read_samples(self, n: int) -> np.ndarray:
        """Return the n most recent samples as [n × n_channels] float32."""
        with self._lock:
            buf = list(self._buffer)
        if len(buf) < n:
            pad = n - len(buf)
            buf = [np.zeros(self.n_channels, dtype=np.float32)] * pad + buf
        return np.array(buf[-n:], dtype=np.float32)


# ─────────────────────────────────────────────
#  REAL DELSYS STREAM via AeroPy
# ─────────────────────────────────────────────
class DelsysAeroPyStream:
    """
    Wraps the AeroPy layer to stream EMG into a rolling numpy buffer.

    Usage:
        stream = DelsysAeroPyStream()
        stream.connect()      # ValidateBase + ScanSensors + Configure
        stream.start()        # Start() + background poll thread
        raw = stream.read_samples(200)
        stream.stop()

    Notes:
    - PollData() returns Dict[Guid, List[double]].
      We capture the EMG channel GUIDs after Configure() (before Start())
      because GUIDs are only available post-configure.
    - Only EMG channels are captured (Type == "EMG" or channel name starts with "EMG").
    - If fewer than N_SENSORS EMG channels are found, the buffer is zero-padded to 6.
    """

    def __init__(self, n_sensors=N_SENSORS):
        self.n_sensors  = n_sensors
        self._lock      = threading.Lock()
        self._buffer    = deque(maxlen=int(EMG_FS * 5))
        self._running   = False
        self._emg_guids = []       # populated after Configure()
        self.TrigBase   = None

    def connect(self):
        """
        Loads DelsysAPI.dll via pythonnet, connects to base,
        scans sensors, and configures the pipeline.
        """
        import clr
        import os

        # Path to DelsysAPI.dll — adjust if your folder structure differs
        dll_path = os.path.join(os.path.dirname(__file__), "..", "resources", "DelsysAPI")
        clr.AddReference(dll_path)
        clr.AddReference("System.Collections")

        # Import AeroPy — this comes from the DelsysAPI.dll once referenced
        from Aero import AeroPy
        from AeroPy.TrignoBase import TrignoBase

        base = TrignoBase()
        self.TrigBase = base.BaseInstance

        print("[Delsys] Connecting to Trigno base...")
        self.TrigBase.ValidateBase(base.key, base.license)
        print("[Delsys] Connected.")

        print("[Delsys] Scanning for sensors...")
        import asyncio
        asyncio.run(self.TrigBase.ScanSensors())

        sensors = self.TrigBase.GetSensors()
        n_found = len(list(sensors))
        print(f"[Delsys] Found {n_found} sensor(s).")

        self.TrigBase.SelectAllSensors()
        self.TrigBase.Configure()
        print("[Delsys] Pipeline configured.")

        # Capture EMG channel GUIDs now (only available post-Configure)
        self._emg_guids = self._get_emg_guids()
        print(f"[Delsys] EMG GUIDs captured: {len(self._emg_guids)} channels")

    def _get_emg_guids(self) -> list:
        """
        Iterates through all sensors and returns GUIDs of enabled EMG channels only.
        EMG channel is always at TrignoChannels index 0 per AeroPy documentation.
        """
        guids = []
        sensors = self.TrigBase.GetSensors()
        for sensor_obj in sensors:
            channels = sensor_obj.TrignoChannels
            for ch in channels:
                name = str(ch.Name)
                enabled = bool(ch.IsEnabled)
                ch_type = str(ch.Type)
                if enabled and ("EMG" in name.upper() or "EMG" in ch_type.upper()):
                    guids.append(ch.Id)
        return guids[:self.n_sensors]   # cap at N_SENSORS

    def start(self):
        self.TrigBase.Start()
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        print("[Delsys] Streaming started.")

    def stop(self):
        self._running = False
        self.TrigBase.Stop()
        print("[Delsys] Streaming stopped.")

    def _poll_loop(self):
        """
        Background thread: polls PollData() whenever new data is ready.
        Assembles per-sample rows from the per-channel lists in the returned dict.
        """
        while self._running:
            if self.TrigBase.CheckDataQueue():
                data_dict = self.TrigBase.PollData()
                self._parse_and_buffer(data_dict)
            time.sleep(POLL_INTERVAL)

    def _parse_and_buffer(self, data_dict):
        """
        data_dict: Dict[Guid, List[double]] from PollData()
        Extracts only the EMG channels (by GUID), aligns them sample-by-sample,
        and pushes rows into the ring buffer.
        """
        # Collect lists for each EMG channel in order
        channel_lists = []
        for guid in self._emg_guids:
            if guid in data_dict:
                channel_lists.append(list(data_dict[guid]))
            else:
                channel_lists.append([])   # sensor not present this poll

        if not channel_lists or not any(channel_lists):
            return

        # Align to shortest non-empty channel
        lengths = [len(c) for c in channel_lists if len(c) > 0]
        if not lengths:
            return
        n_samples = min(lengths)
        if n_samples == 0:
            return

        # Pad missing channels with zeros
        aligned = []
        for ch_list in channel_lists:
            if len(ch_list) >= n_samples:
                aligned.append(ch_list[:n_samples])
            else:
                aligned.append([0.0] * n_samples)

        # Pad to N_SENSORS columns if we have fewer sensors than expected
        while len(aligned) < self.n_sensors:
            aligned.append([0.0] * n_samples)

        # Stack into [n_samples × n_sensors] and push rows
        arr = np.array(aligned, dtype=np.float32).T   # [n_samples, n_sensors]
        with self._lock:
            self._buffer.extend(list(arr))

    def read_samples(self, n: int) -> np.ndarray:
        """Return n most recent samples as [n × n_channels] float32."""
        with self._lock:
            buf = list(self._buffer)
        if len(buf) < n:
            pad = n - len(buf)
            buf = [np.zeros(self.n_sensors, dtype=np.float32)] * pad + buf
        return np.array(buf[-n:], dtype=np.float32)


# ─────────────────────────────────────────────
#  FACTORY
# ─────────────────────────────────────────────
def get_emg_stream():
    if MOCK:
        return MockEMGStream()
    else:
        return DelsysAeroPyStream()
