"""
Check available cameras on the system.

Run this first to find your Sony camera's device ID.
"""

import cv2


def check_cameras(max_id: int = 10):
    """Check which camera IDs are available."""
    print("Checking available cameras...\n")

    available = []

    for i in range(max_id):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()

            print(f"Camera {i}: AVAILABLE")
            print(f"  Resolution: {w}x{h}")
            print(f"  FPS: {fps}")
            print(f"  Backend: {backend}")

            # Try to get a frame
            ret, frame = cap.read()
            if ret:
                print(f"  Frame capture: OK")
            else:
                print(f"  Frame capture: FAILED")

            available.append(i)
            cap.release()
            print()
        else:
            pass  # Skip unavailable cameras silently

    if not available:
        print("No cameras found!")
        print("\nTroubleshooting:")
        print("  - Make sure your Sony camera is connected via USB")
        print("  - Check if it's set to webcam/streaming mode")
        print("  - Try unplugging and reconnecting")
        print("  - Check Device Manager for the camera")
    else:
        print(f"Found {len(available)} camera(s): {available}")
        print(f"\nTo use camera {available[0]} in the demo:")
        print(f"  python vlm_camera_demo.py --camera {available[0]}")


def preview_camera(camera_id: int):
    """Preview a specific camera."""
    print(f"\nPreviewing camera {camera_id}... Press Q to quit.\n")

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Could not open camera {camera_id}")
        return

    # Try to set higher resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Show resolution on frame
        h, w = frame.shape[:2]
        cv2.putText(
            frame, f"Camera {camera_id}: {w}x{h}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
        cv2.putText(
            frame, "Press Q to quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1
        )

        cv2.imshow(f"Camera {camera_id} Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check and preview available cameras")
    parser.add_argument("--preview", type=int, default=None, help="Preview specific camera ID")
    args = parser.parse_args()

    if args.preview is not None:
        preview_camera(args.preview)
    else:
        check_cameras()
