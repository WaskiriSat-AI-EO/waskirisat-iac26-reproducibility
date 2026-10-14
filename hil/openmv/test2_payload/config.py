"""Configuration shared by the OpenMV payload modules.

Target validated by phase 1: OpenMV Cam H7 Plus, OpenMV 5.0.0.
Change wiring, cadence and transfer policy here; do not change model contracts
without re-running the phase-1 validation.
"""

# UART3 pins on H7/H7 Plus: TX=P4, RX=P5. Optional RTS=P1, CTS=P2.
UART_ID = 3
UART_BAUD = 460800
UART_TIMEOUT_MS = 1000
UART_TIMEOUT_CHAR_MS = 100
UART_RX_BUFFER_BYTES = 4096
UART_HARDWARE_FLOW_CONTROL = False

# OBC wake line.
WAKE_PIN_NAME = "P0"
LOW_POWER_ENABLED = False
POST_WAKE_DELAY_MS = 75
SESSION_IDLE_MS = 60000
READY_REPEAT_MS = 1000

# Protocol v2 limits.
PROTOCOL_VERSION = 2
MAX_COMMAND_BYTES = 256
MAX_CONTROL_PAYLOAD_BYTES = 16384
FILE_CHUNK_BYTES = 512
MAX_SOURCE_FILE_BYTES = 8 * 1024 * 1024
DEBUG_ENABLED = True

# Exact phase-1 deployment root and model contracts.
DATA_ROOT_CANDIDATES = (
    "/sdcard/paper_tests",
    "/sd/paper_tests",
    "/paper_tests",
    "paper_tests",
)

THRESHOLD = 0.5
SCHEMA_VERSION = "3.1"
FIRMWARE_VERSION = "openmv-payload-3.1"

MODES = (
    {
        "name": "RGB_CLOUD",
        "dataset": "cloud_detection",
        "model": "models/cloud_detection.tflite",
        "model_size_bytes": 1941072,
        "classes": (("clear", 0), ("cloud", 1)),
        "positive_class": "cloud",
        "input_shape": (1, 224, 224, 3),
        "input_dtype": "b",
        "input_scale": 1.0,
        "input_zero": -128,
        "layout": "NHWC",
        "output_shape": (1, 1),
        "output_dtype": "b",
        "output_scale": 0.00390625,
        "output_zero": -128,
        "output_transform": "probability",
    },
    {
        "name": "THERMAL_FIRE",
        "dataset": "fire_detection",
        "model": "models/fire_detection.tflite",
        "model_size_bytes": 1452472,
        "classes": (("no_fire", 0), ("fire", 1)),
        "positive_class": "fire",
        "input_shape": (1, 3, 120, 160),
        "input_dtype": "f",
        "input_scale": 1.0,
        "input_zero": 0,
        "layout": "NCHW",
        "output_shape": (1, 1),
        "output_dtype": "b",
        "output_scale": 0.04537102,
        "output_zero": -5,
        "output_transform": "sigmoid",
    },
)

# Source files are streamed without loading the whole file into RAM. Set False
# if only inference telemetry is required.
TRANSFER_SOURCE_FILE = True

# Commit state is persisted atomically after the OBC validates each sample.
PERSIST_PROGRESS = True
STATE_FILENAME = "test_2/mission_state_v3"
