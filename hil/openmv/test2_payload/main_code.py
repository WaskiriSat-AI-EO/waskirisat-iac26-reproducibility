import gc
import os
import time
import ujson
import pyb

import config
import model_runner
import payload_protocol as protocol

LED_RED = pyb.LED(1)
LED_GREEN = pyb.LED(2)
LED_BLUE = pyb.LED(3)

_wake_flag = False
_wake_pin = None
_wake_ext = None


def _wake_callback(line):
    global _wake_flag
    _wake_flag = True


def _init_wake():
    global _wake_pin, _wake_ext
    _wake_pin = pyb.Pin(config.WAKE_PIN_NAME, pyb.Pin.IN, pyb.Pin.PULL_DOWN)
    _wake_ext = pyb.ExtInt(
        _wake_pin, pyb.ExtInt.IRQ_RISING, pyb.Pin.PULL_DOWN, _wake_callback
    )


def _led_idle():
    LED_RED.off()
    LED_BLUE.off()
    LED_GREEN.on()


def _led_busy():
    LED_RED.off()
    LED_GREEN.off()
    LED_BLUE.on()


def _led_error():
    LED_GREEN.off()
    LED_BLUE.off()
    LED_RED.on()


def _leds_off():
    LED_RED.off()
    LED_GREEN.off()
    LED_BLUE.off()


def _json_bytes(value):
    return ujson.dumps(value).encode()


def _send_result(sequence, value):
    protocol.send_frame(protocol.MSG_RESULT, sequence, _json_bytes(value))


def _send_ready():
    payload = {
        "firmware_version": config.FIRMWARE_VERSION,
        "protocol_version": config.PROTOCOL_VERSION,
        "available_modes": model_runner.mode_names(),
        "active_mode": model_runner.active_mode()["name"],
        "mem_free": gc.mem_free(),
    }
    protocol.send_frame(protocol.MSG_READY, 0, _json_bytes(payload))


def _mime_type(path):
    lower = path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".bmp"):
        return "image/bmp"
    return "application/octet-stream"


def _send_source_file(sequence, result):
    path = model_runner.reserved_path()
    if not config.TRANSFER_SOURCE_FILE or path is None:
        result["file_transfer"] = {"enabled": False}
        return
    size = int(os.stat(path)[6])
    if size > config.MAX_SOURCE_FILE_BYTES:
        raise ValueError("SOURCE_FILE_TOO_LARGE:%d" % size)
    metadata = {
        "sample_id": result["sample_id"],
        "name": result["image_name"],
        "mime_type": _mime_type(path),
        "size_bytes": size,
        "chunk_bytes": config.FILE_CHUNK_BYTES,
        "kind": "source_image",
    }
    protocol.send_frame(protocol.MSG_FILE_BEGIN, sequence, _json_bytes(metadata))
    started = time.ticks_us()
    whole_crc = 0
    sent = 0
    chunk_index = 0
    with open(path, "rb") as stream:
        while True:
            chunk = stream.read(config.FILE_CHUNK_BYTES)
            if not chunk:
                break
            whole_crc = protocol.crc32(chunk, whole_crc)
            protocol.send_frame(protocol.MSG_FILE_CHUNK, sequence, chunk, chunk_index)
            sent += len(chunk)
            chunk_index += 1
    transfer_us = time.ticks_diff(time.ticks_us(), started)
    footer = {
        "sample_id": result["sample_id"],
        "bytes": sent,
        "chunks": chunk_index,
        "crc32": whole_crc,
        "openmv_uart_tx_us": transfer_us,
    }
    protocol.send_frame(protocol.MSG_FILE_END, sequence, _json_bytes(footer))
    result["file_transfer"] = {
        "enabled": True,
        "kind": "source_image",
        "bytes": sent,
        "chunks": chunk_index,
        "crc32": whole_crc,
        "openmv_uart_tx_us": transfer_us,
    }


def _handle_infer(sequence):
    _led_busy()
    started = time.ticks_us()
    result = model_runner.run_reserved_inference()
    if result is None:
        status = model_runner.status()
        status["no_more_images"] = True
        _send_result(sequence, status)
        protocol.send_done(sequence)
        _led_idle()
        return
    _send_source_file(sequence, result)
    result["metrics"]["transaction_before_result_us"] = time.ticks_diff(
        time.ticks_us(), started
    )
    result["metrics"]["memory_before_result_bytes"] = int(gc.mem_free())
    result["protocol_counters"] = dict(protocol.stats)
    _send_result(sequence, result)
    protocol.send_done(sequence)
    _led_idle()


def _handle_command(frame):
    try:
        command = frame.payload.decode().strip()
    except Exception:
        protocol.send_error(frame.sequence, "COMMAND_ENCODING")
        protocol.send_done(frame.sequence)
        return True
    if len(command) > config.MAX_COMMAND_BYTES:
        protocol.send_error(frame.sequence, "COMMAND_TOO_LONG")
        protocol.send_done(frame.sequence)
        return True

    base = command.split(":", 1)[0]
    protocol.send_ack(frame.sequence, base)
    try:
        if command == "PING":
            return True
        if command == "STATUS":
            value = model_runner.status()
            value["protocol_counters"] = dict(protocol.stats)
            _send_result(frame.sequence, value)
            protocol.send_done(frame.sequence)
            return True
        if command.startswith("SET_MODE:"):
            name = command.split(":", 1)[1].strip()
            changed = model_runner.set_mode(name)
            value = model_runner.status()
            value["mode_changed"] = changed
            _send_result(frame.sequence, value)
            protocol.send_done(frame.sequence)
            return True
        if command == "INFER":
            _handle_infer(frame.sequence)
            return True
        if command.startswith("COMMIT:"):
            sample_id = command.split(":", 1)[1]
            committed = model_runner.commit_sample(sample_id)
            value = model_runner.status()
            value["committed"] = committed
            value["sample_id"] = sample_id
            _send_result(frame.sequence, value)
            protocol.send_done(frame.sequence)
            return True
        if command == "RESET_RUN":
            model_runner.reset_progress()
            model_runner.unload_model()
            value = model_runner.status()
            value["reset_run"] = True
            _send_result(frame.sequence, value)
            protocol.send_done(frame.sequence)
            return True
        if command == "SLEEP":
            protocol.send_done(frame.sequence)
            return False
        protocol.send_error(frame.sequence, "UNKNOWN_COMMAND", command)
        protocol.send_done(frame.sequence)
    except Exception as error:
        gc.collect()
        _led_error()
        protocol.send_error(frame.sequence, base + "_FAIL", repr(error))
        protocol.send_done(frame.sequence)
        _led_idle()
    return True


def _session_loop():
    _led_idle()
    last_activity = time.ticks_ms()
    last_ready = time.ticks_ms() - config.READY_REPEAT_MS
    while True:
        now = time.ticks_ms()
        if time.ticks_diff(now, last_ready) >= config.READY_REPEAT_MS:
            _send_ready()
            last_ready = now
        try:
            frame = protocol.read_frame(timeout_ms=200)
        except protocol.ProtocolError as error:
            # Timeouts are normal while waiting; malformed frames are counted and
            # the magic scanner will resynchronize on the next call.
            if not str(error).endswith("TIMEOUT") and "TIMEOUT:" not in str(error):
                protocol.send_debug(0, "RX_FRAME_ERROR:" + str(error))
            frame = None
        if frame is not None:
            last_activity = time.ticks_ms()
            if frame.msg_type != protocol.MSG_COMMAND:
                protocol.send_error(frame.sequence, "EXPECTED_COMMAND")
            elif not _handle_command(frame):
                return
        if time.ticks_diff(time.ticks_ms(), last_activity) >= config.SESSION_IDLE_MS:
            return
        pyb.wfi()


def _enter_low_power():
    global _wake_flag
    _wake_flag = False
    _leds_off()
    if config.LOW_POWER_ENABLED:
        pyb.stop()
    else:
        while not _wake_flag:
            pyb.delay(10)


def main():
    protocol.init_uart()
    _init_wake()
    try:
        model_runner.initialize()
    except Exception as error:
        _led_error()
        protocol.send_error(0, "INIT_FAIL", repr(error))
        raise
    gc.collect()
    # Cold boot accepts commands immediately; subsequent sessions start on wake.
    while True:
        try:
            _session_loop()
        except Exception as error:
            gc.collect()
            _led_error()
            try:
                protocol.send_error(0, "SESSION_FAIL", repr(error))
            except Exception:
                pass
        _enter_low_power()
        pyb.delay(config.POST_WAKE_DELAY_MS)
        protocol.init_uart()
        protocol.flush_rx(10)


main()
