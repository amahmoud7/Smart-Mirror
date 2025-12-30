"""
Convert ISIC DICOM files to JPEG format.

Usage:
    python convert_dicom_to_jpeg.py --input-dir /path/to/dicom --output-dir /path/to/jpeg
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Progress counter
progress_lock = threading.Lock()
converted_count = 0
error_count = 0


def convert_single_file(args):
    """Convert a single DICOM file to JPEG."""
    global converted_count, error_count

    input_path, output_path = args

    try:
        import pydicom
        from PIL import Image
        import numpy as np

        # Read DICOM
        dcm = pydicom.dcmread(input_path)

        # Get pixel array
        pixel_array = dcm.pixel_array

        # Normalize to 0-255
        if pixel_array.max() > 255:
            pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
        else:
            pixel_array = pixel_array.astype(np.uint8)

        # Handle different image formats
        if len(pixel_array.shape) == 2:
            # Grayscale
            img = Image.fromarray(pixel_array, mode='L').convert('RGB')
        elif len(pixel_array.shape) == 3:
            if pixel_array.shape[2] == 3:
                # RGB
                img = Image.fromarray(pixel_array, mode='RGB')
            elif pixel_array.shape[2] == 4:
                # RGBA
                img = Image.fromarray(pixel_array, mode='RGBA').convert('RGB')
            else:
                # Other multi-channel, take first 3
                img = Image.fromarray(pixel_array[:, :, :3], mode='RGB')
        else:
            raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")

        # Save as JPEG
        img.save(output_path, 'JPEG', quality=95)

        with progress_lock:
            converted_count += 1
            if converted_count % 500 == 0:
                print(f"  Converted: {converted_count}")

        return True, input_path

    except Exception as e:
        with progress_lock:
            error_count += 1
        return False, f"{input_path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Convert DICOM to JPEG")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory with .dcm files")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for .jpg files")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    # Check dependencies
    try:
        import pydicom
        from PIL import Image
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pydicom pillow")
        return

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all DICOM files
    dcm_files = list(input_dir.glob("*.dcm"))
    print(f"Found {len(dcm_files)} DICOM files")

    if not dcm_files:
        print("No .dcm files found!")
        return

    # Prepare conversion tasks
    tasks = []
    for dcm_path in dcm_files:
        jpg_name = dcm_path.stem + ".jpg"
        jpg_path = output_dir / jpg_name

        # Skip if already converted
        if not jpg_path.exists():
            tasks.append((str(dcm_path), str(jpg_path)))

    print(f"Files to convert: {len(tasks)} (skipping {len(dcm_files) - len(tasks)} already done)")

    if not tasks:
        print("All files already converted!")
        return

    # Convert in parallel
    print(f"\nConverting with {args.workers} workers...")

    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(convert_single_file, task): task for task in tasks}

        for future in as_completed(futures):
            success, result = future.result()
            if not success:
                errors.append(result)

    print(f"\n{'=' * 50}")
    print(f"Conversion complete!")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Errors: {error_count}")

    if errors and len(errors) <= 10:
        print("\nErrors:")
        for err in errors:
            print(f"  {err}")
    elif errors:
        print(f"\nFirst 10 errors:")
        for err in errors[:10]:
            print(f"  {err}")

    print(f"\nJPEG files saved to: {output_dir}")


if __name__ == "__main__":
    main()
