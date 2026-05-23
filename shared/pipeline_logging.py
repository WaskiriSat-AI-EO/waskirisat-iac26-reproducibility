"""Shared logging configuration for nina_thermal pipeline scripts.

Sets up two handlers for every pipeline run:
- StreamHandler  — terminal output (INFO level, short format)
- FileHandler    — persistent .log file (DEBUG level, verbose format)

The log file receives a structured header (stage, ISO start time, CLI args)
and the caller should write a footer with elapsed time via
:func:`log_run_footer`.

Usage::

    from nina_thermal.pipelines._logging import configure_logging, log_run_footer

    log_file, start_time = configure_logging(
        "run_extraction",
        log_dir="data/logs",
    )
    ...
    return log_run_footer(start_time, exit_code=0)

Grouping stages into a single run
----------------------------------
Pass a shared ``--log_dir`` to all pipeline scripts to keep every stage's
log file together in one directory::

    --log_dir data/logs/run_20260405_223133

Each stage still produces its own ``.log`` file inside that directory, which
is the correct granularity: stages run at different wall-clock times and may
be re-run independently (e.g. re-simulate with new profiles without
re-acquiring data).
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple


_STREAM_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
_FILE_FORMAT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_FILE_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_SEPARATOR = "=" * 72


def configure_logging(
    stage_name: str,
    log_dir: str = "data/logs",
    extra_id: Optional[str] = None,
    argv: Optional[list] = None,
) -> Tuple[Path, datetime]:
    """Configure root logger with terminal + file handlers.

    Should be called once at the start of each pipeline script, before any
    other logging calls.  Replaces ``logging.basicConfig()``.

    Args:
        stage_name: Pipeline stage name used in the log filename
            (e.g. ``"run_extraction"``).
        log_dir:    Directory for log files.  Created automatically if absent.
            Pass a shared path across all stages of one run to group their
            log files together (e.g. ``"data/logs/run_20260405_223133"``).
        extra_id:   Optional extra identifier appended to the filename before
            the timestamp, e.g. ``"20230101-20230630"`` for the acquisition
            stage so the sensing window is visible in the filename.
        argv:       ``sys.argv[1:]`` or equivalent, written verbatim into the
            log header.  If ``None``, the header omits the CLI args line.

    Returns:
        ``(log_file, start_time)`` — path to the opened log file and the
        UTC datetime when logging was initialised (pass ``start_time`` to
        :func:`log_run_footer`).
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    start_time = datetime.now(timezone.utc)
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")

    parts = [stage_name]
    if extra_id:
        parts.append(extra_id)
    parts.append(timestamp)
    log_file = log_path / f"{'_'.join(parts)}.log"

    # ── Write structured header directly to file ──────────────────────────────
    with open(log_file, "w", encoding="utf-8") as fh:
        fh.write(f"{_SEPARATOR}\n")
        fh.write(f"  STAGE   : {stage_name}\n")
        fh.write(f"  STARTED : {start_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        if argv is not None:
            fh.write(f"  ARGS    : {' '.join(str(a) for a in argv)}\n")
        fh.write(f"{_SEPARATOR}\n\n")

    # ── Configure root logger ─────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_STREAM_FORMAT, datefmt=_DATE_FORMAT))

    # Append mode — header was already written above
    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATE_FORMAT)
    )

    root.addHandler(stream_handler)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info("Log → %s", log_file)
    return log_file, start_time


def log_run_footer(start_time: datetime, exit_code: int = 0) -> int:
    """Write elapsed time and exit code to the log, then return ``exit_code``.

    Designed to be used as a pass-through in pipeline ``main()`` functions::

        return log_run_footer(start_time, exit_code=result)

    Args:
        start_time: Value returned by :func:`configure_logging`.
        exit_code:  Integer exit code to record (0 = success).

    Returns:
        ``exit_code`` unchanged, for convenient ``return`` chaining.
    """
    elapsed = datetime.now(timezone.utc) - start_time
    total_seconds = int(elapsed.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    elapsed_str = f"{hours:02d}h {minutes:02d}m {seconds:02d}s"

    logger = logging.getLogger(__name__)
    logger.info("Run finished — exit_code=%d  elapsed=%s", exit_code, elapsed_str)

    # Append footer to log file via the file handler
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            try:
                if handler.stream is None:
                    break
                handler.stream.write(
                    f"\n{_SEPARATOR}\n"
                    f"  FINISHED: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
                    f"  ELAPSED : {elapsed_str}\n"
                    f"  EXIT    : {exit_code}\n"
                    f"{_SEPARATOR}\n"
                )
                handler.stream.flush()
            except Exception:
                pass
            break

    return exit_code
