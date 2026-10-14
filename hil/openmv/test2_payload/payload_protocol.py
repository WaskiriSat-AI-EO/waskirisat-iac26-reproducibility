"""CRC-protected, sequence-aware UART framing for OpenMV.

Wire header (20 bytes, little endian):
  magic[2], version:u8, type:u8, sequence:u16, flags:u16,
  payload_length:u32, payload_crc32:u32, header_crc32:u32
"""

import binascii
import time
import ustruct
import pyb

import config

MAGIC = b"OM"
HEADER_PREFIX_FORMAT = "<2sBBHHII"
HEADER_FORMAT = "<2sBBHHIII"
HEADER_PREFIX_SIZE = 16
HEADER_SIZE = 20

MSG_READY = 0x01
MSG_ACK = 0x02
MSG_RESULT = 0x04
MSG_ERROR = 0x05
MSG_DONE = 0x06
MSG_DEBUG = 0x07
MSG_COMMAND = 0x10
MSG_FILE_BEGIN = 0x20
MSG_FILE_CHUNK = 0x21
MSG_FILE_END = 0x22

FLAG_NONE = 0


class ProtocolError(Exception):
    pass


class Frame:
    def __init__(self, msg_type, sequence, flags, payload):
        self.msg_type = msg_type
        self.sequence = sequence
        self.flags = flags
        self.payload = payload


uart = None
stats = {
    "rx_frames": 0,
    "tx_frames": 0,
    "rx_bytes": 0,
    "tx_bytes": 0,
    "header_crc_errors": 0,
    "payload_crc_errors": 0,
    "version_errors": 0,
    "length_errors": 0,
    "rx_timeouts": 0,
}


def crc32(data, value=0):
    return int(binascii.crc32(data, value)) & 0xFFFFFFFF


def init_uart():
    global uart
    flow = 0
    if config.UART_HARDWARE_FLOW_CONTROL:
        flow = pyb.UART.RTS | pyb.UART.CTS
    uart = pyb.UART(
        config.UART_ID,
        config.UART_BAUD,
        timeout=config.UART_TIMEOUT_MS,
        timeout_char=config.UART_TIMEOUT_CHAR_MS,
        read_buf_len=config.UART_RX_BUFFER_BYTES,
        flow=flow,
    )
    return uart


def flush_rx(quiet_ms=20):
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < quiet_ms:
        if uart is not None and uart.any():
            uart.read()
            start = time.ticks_ms()
        else:
            pyb.delay(1)


def _write_all(data):
    if uart is None:
        raise ProtocolError("UART_NOT_INITIALIZED")
    offset = 0
    while offset < len(data):
        written = uart.write(data[offset:])
        if written is None:
            written = len(data) - offset
        if written <= 0:
            pyb.delay(1)
            continue
        offset += written
    stats["tx_bytes"] += len(data)


def encode_header(msg_type, sequence, flags, payload_len, payload_crc):
    prefix = ustruct.pack(
        HEADER_PREFIX_FORMAT,
        MAGIC,
        config.PROTOCOL_VERSION,
        msg_type,
        sequence & 0xFFFF,
        flags & 0xFFFF,
        payload_len,
        payload_crc,
    )
    return prefix + ustruct.pack("<I", crc32(prefix))


def send_frame(msg_type, sequence, payload=b"", flags=FLAG_NONE):
    if payload is None:
        payload = b""
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        payload = str(payload).encode()
    header = encode_header(msg_type, sequence, flags, len(payload), crc32(payload))
    _write_all(header)
    if payload:
        _write_all(payload)
    stats["tx_frames"] += 1


def _read_exact(length, timeout_ms):
    data = bytearray(length)
    view = memoryview(data)
    offset = 0
    last_progress = time.ticks_ms()
    while offset < length:
        if uart is not None and uart.any():
            count = uart.readinto(view[offset:])
            if count:
                offset += count
                last_progress = time.ticks_ms()
                continue
        if time.ticks_diff(time.ticks_ms(), last_progress) >= timeout_ms:
            stats["rx_timeouts"] += 1
            raise ProtocolError("RX_TIMEOUT:%d/%d" % (offset, length))
        pyb.wfi()
    stats["rx_bytes"] += length
    return bytes(data)


def _read_magic(timeout_ms):
    start = time.ticks_ms()
    got_o = False
    while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
        if uart is not None and uart.any():
            raw = uart.read(1)
            if not raw:
                continue
            stats["rx_bytes"] += 1
            value = raw[0]
            if not got_o:
                got_o = value == 0x4F
            elif value == 0x4D:
                return
            else:
                got_o = value == 0x4F
        else:
            pyb.wfi()
    raise ProtocolError("MAGIC_TIMEOUT")


def read_frame(timeout_ms=500, max_payload=None):
    if max_payload is None:
        max_payload = config.MAX_CONTROL_PAYLOAD_BYTES
    _read_magic(timeout_ms)
    tail = _read_exact(HEADER_SIZE - 2, timeout_ms)
    header = MAGIC + tail
    unpacked = ustruct.unpack(HEADER_FORMAT, header)
    _, version, msg_type, sequence, flags, length, payload_crc, header_crc = unpacked

    if crc32(header[:HEADER_PREFIX_SIZE]) != header_crc:
        stats["header_crc_errors"] += 1
        raise ProtocolError("HEADER_CRC")
    if version != config.PROTOCOL_VERSION:
        stats["version_errors"] += 1
        raise ProtocolError("VERSION:%d" % version)
    if length > max_payload:
        stats["length_errors"] += 1
        raise ProtocolError("PAYLOAD_TOO_LARGE:%d" % length)

    payload = _read_exact(length, timeout_ms) if length else b""
    if crc32(payload) != payload_crc:
        stats["payload_crc_errors"] += 1
        raise ProtocolError("PAYLOAD_CRC")
    stats["rx_frames"] += 1
    return Frame(msg_type, sequence, flags, payload)


def send_text(msg_type, sequence, text, flags=FLAG_NONE):
    send_frame(msg_type, sequence, text.encode() if isinstance(text, str) else text, flags)


def send_ack(sequence, command):
    send_text(MSG_ACK, sequence, command)


def send_error(sequence, code, detail=""):
    text = code if not detail else code + ":" + str(detail)
    send_text(MSG_ERROR, sequence, text)


def send_done(sequence):
    send_frame(MSG_DONE, sequence)


def send_debug(sequence, text):
    if config.DEBUG_ENABLED:
        send_text(MSG_DEBUG, sequence, text)
