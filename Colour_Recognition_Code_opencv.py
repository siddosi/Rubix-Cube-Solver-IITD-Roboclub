'''import cv2
import numpy as np

# HSV color ranges for Rubik's cube colors
# Format: (lower_hsv, upper_hsv)
HSV_COLOR_RANGES = {
    "red1": ([0, 100, 100], [10, 255, 255]),     # Red spans across 0, so we need two ranges
    "red2": ([160, 100, 100], [180, 255, 255]),  # Second red range
    "blue": ([100, 100, 100], [130, 255, 255]),
    "green": ([40, 100, 100], [80, 255, 255]),
    "white": ([0, 0, 200], [180, 30, 255]),
    "yellow": ([20, 100, 100], [40, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255])
}

def classify_color_hsv(hsv_pixel):
    """Classify a pixel's HSV values to a named color"""
    h, s, v = hsv_pixel

    # Check special case for red (wraps around 180/0)
    if ((0 <= h <= 10) or (160 <= h <= 180)) and s >= 100 and v >= 100:
        if h <= 10:  # Lower red range
            return "red"
        else:  # Upper red range
            return "red"

    # Check other colors
    for name, (lower, upper) in HSV_COLOR_RANGES.items():
        if name == "red1" or name == "red2":  # Skip red as we handled it above
            continue

        if all(lower[i] <= hsv_pixel[i] <= upper[i] for i in range(3)):
            return name

    return "unknown"

def get_dominant_color(region):
    """Find the most common color in a region using a histogram approach"""
    # Convert to HSV
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Flatten the HSV values
    pixels = hsv_region.reshape(-1, 3)

    # Create bins for counting colors
    color_counts = {color: 0 for color in ["red", "blue", "green", "white", "yellow", "orange", "unknown"]}

    # Count pixels for each color
    for pixel in pixels:
        color = classify_color_hsv(pixel)
        color_counts[color] += 1

    # Find the most common color
    max_color = max(color_counts, key=color_counts.get)

    # If most pixels are unknown, use the average HSV as a fallback
    if max_color == "unknown" or color_counts[max_color] < len(pixels) * 0.3:
        avg_hsv = np.mean(pixels, axis=0).astype(int)
        max_color = classify_color_hsv(avg_hsv)

    return max_color

def detect_rubiks_cube_colors(image_path, debug=False):
    """Detect Rubik's cube colors from an image"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded. Check path.")

    # Resize for consistent processing
    img = cv2.resize(img, (300, 300))
    h, w = img.shape[:2]

    # Calculate the size of each cell
    step_h, step_w = h // 3, w // 3

    # Add a small margin to avoid edges between stickers
    margin = 5

    cube_matrix = []
    debug_img = img.copy() if debug else None

    for i in range(3):
        row = []
        for j in range(3):
            # Calculate region with margin
            y1, y2 = i * step_h + margin, (i + 1) * step_h - margin
            x1, x2 = j * step_w + margin, (j + 1) * step_w - margin

            # Extract region
            region = img[y1:y2, x1:x2]

            # Get dominant color
            color_name = get_dominant_color(region)
            row.append(color_name)

            # Draw rectangle for debugging
            if debug:
                color_bgr = {
                    "red": (0, 0, 255),
                    "blue": (255, 0, 0),
                    "green": (0, 255, 0),
                    "white": (255, 255, 255),
                    "yellow": (0, 255, 255),
                    "orange": (0, 140, 255),
                    "unknown": (128, 128, 128)
                }

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color_bgr.get(color_name, (128, 128, 128)), 2)
                cv2.putText(debug_img, color_name, (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cube_matrix.append(row)

    # Save debug image if requested
    if debug:
        cv2.imwrite("debug_cube.jpg", debug_img)

    return cube_matrix

# Example usage
if __name__ == "__main__":
    image_path = "/content/Screenshot 2025-05-20 231437.png"  # Replace with your image path

    # Set debug=True to see visual output of detected colors
    result = detect_rubiks_cube_colors(image_path, debug=True)

    print("Detected 3x3 Rubik's Cube Colors:")
    for row in result:
        print(row)'''





'''import cv2
import numpy as np



# Assuming that the rubiks cube to be solved has the following six colours : RGBWOY
# Defining dictionary for the colours, data may be changed as per the cubes colour intensity and lighting effects


Hsv_range = {
    "red1": ([0, 100, 100], [10, 255, 255]),
    "red2": ([160, 100, 100], [180, 255, 255]),
    "blue": ([100, 100, 100], [130, 255, 255]),
    "green": ([40, 100, 100], [80, 255, 255]),
    "white": ([0, 0, 200], [180, 30, 255]),
    "yellow": ([20, 100, 100], [40, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255])
}







def classify_color_hsv(hsv_pixel):
    """Classify a pixel's HSV values to a named color"""
    h, s, v = hsv_pixel

    # Check special case for red (wraps around 180/0)
    if ((0 <= h <= 10) or (160 <= h <= 180)) and s >= 100 and v >= 100:
        return "red"

    # Check other colors
    for name in Hsv_range:
        if name == "red1" or name == "red2":  # Skip red as we handled it above
            continue

        lower, upper = Hsv_range[name]

        match = True
        for i in range(3):
            if not (lower[i] <= hsv_pixel[i] <= upper[i]):
                match = False
                break

        if match:
            return name

    return "unknown"

def get_dominant_color(region):
    """Find the most common color in a region using a histogram approach"""
    # Convert to HSV
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Flatten the HSV values
    pixels = hsv_region.reshape(-1, 3)

    # Create bins for counting colors
    color_counts = {color: 0 for color in ["red", "blue", "green", "white", "yellow", "orange", "unknown"]}

    # Count pixels for each color
    for pixel in pixels:
        color = classify_color_hsv(pixel)
        color_counts[color] += 1

    # Find the most common color
    max_color = max(color_counts, key=color_counts.get)

    # If most pixels are unknown, use the average HSV as a fallback
    if max_color == "unknown" or color_counts[max_color] < len(pixels) * 0.3:
        avg_hsv = np.mean(pixels, axis=0).astype(int)
        max_color = classify_color_hsv(avg_hsv)

    return max_color




def detect_rubiks_cube_colors(image_path, debug=False):
    """Detect Rubik's cube colors from an image"""
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("NO IMAGE COULD BE LOADED. CHECK PATH.")

    # Resize for consistent processing
    img = cv2.resize(img, (300, 300))
    h, w = img.shape[:2]

    # Calculate the size of each cell
    step_h, step_w = h // 3, w // 3

    # Add a small margin to avoid edges between stickers
    margin = 5

    cube_matrix = []
    debug_img = img.copy() if debug else None

    for i in range(3):
        row = []
        for j in range(3):
            # Calculate region with margin
            y1, y2 = i * step_h + margin, (i + 1) * step_h - margin
            x1, x2 = j * step_w + margin, (j + 1) * step_w - margin

            # Extract region
            region = img[y1:y2, x1:x2]

            # Get dominant color
            color_name = get_dominant_color(region)
            row.append(color_name)

            # Draw rectangle for debugging
            if debug:
                color_bgr = {
                    "red": (0, 0, 255),
                    "blue": (255, 0, 0),
                    "green": (0, 255, 0),
                    "white": (255, 255, 255),
                    "yellow": (0, 255, 255),
                    "orange": (0, 140, 255),
                    "unknown": (128, 128, 128)
                }

                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color_bgr.get(color_name, (128, 128, 128)), 2)
                cv2.putText(debug_img, color_name, (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cube_matrix.append(row)

    # Save debug image if requested
    if debug:
        cv2.imwrite("debug_cube.jpg", debug_img)
    return cube_matrix





image_path = input()              #"/content/Screenshot 2025-05-20 231437.png"

    # Set debug=True to see visual output of detected colors
result = detect_rubiks_cube_colors(image_path, debug=True)

print("Detected 3x3 Rubik's Cube Colors:")
for row in result:
    print(row)'''






!pip install opencv-python numpy
import cv2
import numpy as np

Hsv_range = {
    "red1": ([0, 100, 100], [10, 255, 255]),
    "red2": ([160, 100, 100], [180, 255, 255]),
    "blue": ([100, 100, 100], [130, 255, 255]),
    "green": ([40, 100, 100], [80, 255, 255]),
    "white": ([0, 0, 200], [180, 30, 255]),
    "yellow": ([20, 100, 100], [40, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255])
}

def classify_color_hsv(hsv_pixel):
    h, s, v = hsv_pixel
    if ((0 <= h <= 10) or (160 <= h <= 180)) and s >= 100 and v >= 100:
        return "red"
    for name in Hsv_range:
        if name == "red1" or name == "red2":
            continue
        lower, upper = Hsv_range[name]
        match = True
        for i in range(3):
            if not (lower[i] <= hsv_pixel[i] <= upper[i]):
                match = False
                break
        if match:
            return name
    return "unknown"

def get_dominant_color(region):
    hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    pixels = hsv_region.reshape(-1, 3)
    color_counts = {color: 0 for color in ["red", "blue", "green", "white", "yellow", "orange", "unknown"]}
    for pixel in pixels:
        color = classify_color_hsv(pixel)
        color_counts[color] += 1
    max_color = max(color_counts, key=color_counts.get)
    if max_color == "unknown" or color_counts[max_color] < len(pixels) * 0.3:
        avg_hsv = np.mean(pixels, axis=0).astype(int)
        max_color = classify_color_hsv(avg_hsv)
    return max_color

def detect_rubiks_cube_colors(image_path, debug=False):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("NO IMAGE COULD BE LOADED. CHECK PATH.")
    img = cv2.resize(img, (300, 300))
    h, w = img.shape[:2]
    step_h, step_w = h // 3, w // 3
    margin = 5
    cube_matrix = []
    debug_img = img.copy() if debug else None
    for i in range(3):
        row = []
        for j in range(3):
            y1, y2 = i * step_h + margin, (i + 1) * step_h - margin
            x1, x2 = j * step_w + margin, (j + 1) * step_w - margin
            region = img[y1:y2, x1:x2]
            color_name = get_dominant_color(region)
            row.append(color_name)
            if debug:
                color_bgr = {
                    "red": (0, 0, 255),
                    "blue": (255, 0, 0),
                    "green": (0, 255, 0),
                    "white": (255, 255, 255),
                    "yellow": (0, 255, 255),
                    "orange": (0, 140, 255),
                    "unknown": (128, 128, 128)
                }
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), color_bgr.get(color_name, (128, 128, 128)), 2)
                cv2.putText(debug_img, color_name, (x1 + 5, y1 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cube_matrix.append(row)
    if debug:
        cv2.imwrite("debug_cube.jpg", debug_img)
    return cube_matrix

image_path = '/content/Screenshot 2025-05-25 200838.png'
result = detect_rubiks_cube_colors(image_path, debug=True)


for row in result:
    print(*row)
