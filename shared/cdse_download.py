"""CDSE product download and .zip extraction.

Downloads a single product ZIP from the Copernicus Data Space Ecosystem
zipper service, streams it to disk with progress reporting, extracts
the archive, and optionally deletes the ZIP afterwards.
"""

import logging
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DOWNLOAD_BASE = (
    "https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
)
_CHUNK_BYTES = 8192
_TIMEOUT_S   = 600   # large granules can be several hundred MB


def download_and_extract(
    access_token: str,
    product_id: str,
    product_name: str,
    output_dir: Path,
    delete_zip: bool = True,
) -> bool:
    """Download a CDSE product ZIP, extract it, and optionally delete the ZIP.

    Streams the download in ``_CHUNK_BYTES``-sized chunks and displays a
    ``tqdm`` progress bar when the server provides a ``content-length``
    header.  If the destination ``.zip`` already exists it is not
    re-downloaded (idempotent resume behaviour).

    The extracted ``.SEN3`` directory lands directly in *output_dir*
    alongside the ZIP (or in its place if *delete_zip* is ``True``).

    Args:
        access_token: Bearer token from :func:`get_access_token`.
        product_id: Product UUID (``"Id"`` field from search results).
        product_name: Human-readable product name used for the ZIP
            filename and log messages.
        output_dir: Directory in which to save and extract the product.
            Created if it does not exist.
        delete_zip: If ``True`` (default), delete the ``.zip`` after
            successful extraction.

    Returns:
        ``True`` if the product was successfully downloaded and extracted,
        ``False`` on any failure.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / f"{product_name}.zip"

    # ── Download ──────────────────────────────────────────────────────────────
    if not zip_path.exists():
        url = _DOWNLOAD_BASE.format(product_id=product_id)
        headers = {"Authorization": f"Bearer {access_token}"}

        logger.info("Downloading %s …", product_name)
        try:
            response = requests.get(
                url, headers=headers, stream=True, timeout=_TIMEOUT_S
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error(
                "Download HTTP error %s for %s: %s",
                exc.response.status_code, product_name, exc,
            )
            return False
        except Exception as exc:
            logger.error("Download failed for %s: %s", product_name, exc)
            return False

        total = int(response.headers.get("content-length", 0)) or None
        try:
            with open(zip_path, "wb") as fh, tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=product_name[-40:],
                leave=False,
            ) as bar:
                for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                    if chunk:
                        fh.write(chunk)
                        bar.update(len(chunk))
        except Exception as exc:
            logger.error("Write error for %s: %s", product_name, exc)
            zip_path.unlink(missing_ok=True)
            return False

        logger.info("Download complete: %s", zip_path.name)
    else:
        logger.info("ZIP already exists, skipping download: %s", zip_path.name)

    # ── Extract ───────────────────────────────────────────────────────────────
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
        logger.info("Extracted %s → %s", zip_path.name, output_dir)
    except zipfile.BadZipFile as exc:
        logger.error("Bad ZIP for %s: %s", product_name, exc)
        zip_path.unlink(missing_ok=True)
        return False
    except Exception as exc:
        logger.error("Extraction failed for %s: %s", product_name, exc)
        return False

    # ── Optional cleanup ──────────────────────────────────────────────────────
    if delete_zip:
        try:
            zip_path.unlink()
            logger.debug("Deleted ZIP: %s", zip_path.name)
        except Exception as exc:
            logger.warning("Could not delete ZIP %s: %s", zip_path.name, exc)

    return True
