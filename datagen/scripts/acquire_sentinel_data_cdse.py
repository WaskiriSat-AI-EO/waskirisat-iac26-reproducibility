import os
import sys
import requests
import json
from datetime import datetime
from dotenv import load_dotenv
from sentinelsat import read_geojson, geojson_to_wkt

# Load environment variables from .env file
load_dotenv()


def get_access_token(username, password):
    """
    Obtain an access token from the Copernicus Data Space Ecosystem.

    Args:
        username (str): CDSE username
        password (str): CDSE password

    Returns:
        str: Access token
    """
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

    data = {
        "client_id": "cdse-public",
        "username": username,
        "password": password,
        "grant_type": "password",
    }

    try:
        response = requests.post(token_url, data=data, timeout=30)
        response.raise_for_status()
        return response.json()["access_token"]
    except Exception as e:
        print(f"[ERROR] Failed to obtain access token: {e}", file=sys.stderr)
        return None


def search_products(
    access_token, footprint_wkt, start_date, end_date, max_cloud_cover, max_results=100
):
    """
    Search for Sentinel-2 products in the CDSE catalog.

    Args:
        access_token (str): OAuth2 access token
        footprint_wkt (str): WKT representation of the area of interest
        start_date (str): Start date in 'YYYYMMDD' format
        end_date (str): End date in 'YYYYMMDD' format
        max_cloud_cover (int): Maximum cloud cover percentage
        max_results (int): Maximum number of results to return

    Returns:
        list: List of product metadata dictionaries
    """
    # Convert date format from YYYYMMDD to YYYY-MM-DD
    start = datetime.strptime(start_date, "%Y%m%d").strftime("%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y%m%d").strftime("%Y-%m-%d")

    search_url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    # Build OData filter query
    filter_query = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/OData.CSC.DoubleAttribute/Value le {max_cloud_cover}) and "
        f"ContentDate/Start ge {start}T00:00:00.000Z and "
        f"ContentDate/Start le {end}T23:59:59.999Z and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint_wkt}')"
    )

    params = {
        "$filter": filter_query,
        "$top": max_results,
        "$orderby": "ContentDate/Start desc",
    }

    headers = {"Authorization": f"Bearer {access_token}"}

    try:
        response = requests.get(search_url, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("value", [])
    except Exception as e:
        print(f"[ERROR] Search failed: {e}", file=sys.stderr)
        return []


def download_product(access_token, product_id, product_name, output_dir):
    """
    Download a single product from CDSE.

    Args:
        access_token (str): OAuth2 access token
        product_id (str): Product UUID
        product_name (str): Product name
        output_dir (str): Directory to save the product

    Returns:
        bool: True if download was successful
    """
    download_url = (
        f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    )

    headers = {"Authorization": f"Bearer {access_token}"}

    output_path = os.path.join(output_dir, f"{product_name}.zip")

    # Check if file already exists
    if os.path.exists(output_path):
        print(f"[*] Product {product_name} already exists, skipping.")
        return True

    try:
        print(f"[*] Downloading {product_name}...")
        response = requests.get(download_url, headers=headers, stream=True, timeout=300)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))
        block_size = 8192

        with open(output_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r[*] Progress: {percent:.1f}%", end="", flush=True)

        print(f"\n[+] Successfully downloaded {product_name}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to download {product_name}: {e}", file=sys.stderr)
        # Clean up partial download
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def acquire_sentinel_data(
    username,
    password,
    roi_geojson_path,
    start_date,
    end_date,
    max_cloud_cover,
    output_dir,
    max_products=5,
):
    """
    Main function to search and download Sentinel-2 data from CDSE.

    Args:
        username (str): CDSE username
        password (str): CDSE password
        roi_geojson_path (str): Path to the GeoJSON file defining the ROI
        start_date (str): Start date in 'YYYYMMDD' format
        end_date (str): End date in 'YYYYMMDD' format
        max_cloud_cover (int): Maximum cloud cover percentage (0-100)
        output_dir (str): Directory where data will be downloaded
        max_products (int): Maximum number of products to download

    Returns:
        list: List of downloaded product names
    """
    print(f"[*] Starting acquisition for ROI: {roi_geojson_path}")
    print(f"[*] Date range: {start_date} to {end_date}")
    print(f"[*] Max cloud cover: {max_cloud_cover}%")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get access token
    print("[*] Authenticating with CDSE...")
    access_token = get_access_token(username, password)
    if not access_token:
        return None
    print("[+] Authentication successful")

    # Read and convert GeoJSON to WKT
    try:
        footprint_wkt = geojson_to_wkt(read_geojson(roi_geojson_path))
    except Exception as e:
        print(f"[ERROR] Failed to read GeoJSON: {e}", file=sys.stderr)
        return None

    # Search for products
    print("[*] Searching for products...")
    products = search_products(
        access_token, footprint_wkt, start_date, end_date, max_cloud_cover
    )

    if not products:
        print("[!] No products found for the given criteria.")
        return None

    print(f"[+] Found {len(products)} products")

    # Limit the number of products to download
    products_to_download = products[:max_products]
    if len(products) > max_products:
        print(f"[*] Limiting download to first {max_products} products")

    # Download products
    downloaded = []
    for i, product in enumerate(products_to_download, 1):
        product_id = product["Id"]
        product_name = product["Name"]

        print(f"\n[*] Product {i}/{len(products_to_download)}: {product_name}")

        # Display product metadata
        if "ContentDate" in product:
            print(f"    Date: {product['ContentDate'].get('Start', 'N/A')}")

        # Find cloud cover in attributes
        cloud_cover = "N/A"
        if "Attributes" in product:
            for attr in product.get("Attributes", []):
                if attr.get("Name") == "cloudCover":
                    cloud_cover = f"{attr.get('Value', 'N/A')}%"
                    break
        print(f"    Cloud cover: {cloud_cover}")

        if download_product(access_token, product_id, product_name, output_dir):
            downloaded.append(product_name)

    print(
        f"\n[*] Acquisition complete. Downloaded {len(downloaded)}/{len(products_to_download)} products."
    )
    print(f"[*] Data saved in: {output_dir}")

    return downloaded


if __name__ == "__main__":
    # Load credentials from environment
    CDSE_USER = os.getenv("CDSE_USER")
    CDSE_PASSWORD = os.getenv("CDSE_PASSWORD")

    if not CDSE_USER or not CDSE_PASSWORD:
        print("[ERROR] Please set CDSE_USER and CDSE_PASSWORD in your .env file.")
        print("        Register at: https://dataspace.copernicus.eu/")
        sys.exit(1)

    ROI_FILE = "./datagen/resources/bolivia_roi.geojson"
    OUTPUT_DIRECTORY = "./datagen/sentinel/sentinel_data_raw"

    # Create a dummy GeoJSON file if it doesn't exist
    if not os.path.exists(ROI_FILE):
        print(f"[*] Creating example GeoJSON file: {ROI_FILE}")
        with open(ROI_FILE, "w") as f:
            json.dump(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [-66.2, -17.4],
                                [-66.1, -17.4],
                                [-66.1, -17.3],
                                [-66.2, -17.3],
                                [-66.2, -17.4],
                            ]
                        ],
                    },
                },
                f,
                indent=2,
            )

    # Run acquisition
    acquire_sentinel_data(
        username=CDSE_USER,
        password=CDSE_PASSWORD,
        roi_geojson_path=ROI_FILE,
        start_date="20240501",
        end_date="20240531",
        max_cloud_cover=30,
        output_dir=OUTPUT_DIRECTORY,
        max_products=2,
    )
