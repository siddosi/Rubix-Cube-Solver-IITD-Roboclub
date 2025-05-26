import kociemba
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

result_1 = detect_rubiks_cube_colors(input('image_path_1'), debug = True)
result_2 = detect_rubiks_cube_colors(input('image_path_2'), debug = True)
result_3 = detect_rubiks_cube_colors(input('image_path_3'), debug= True)
result_4 = detect_rubiks_cube_colors(input('image_path_4'), debug = True)
result_5 = detect_rubiks_cube_colors(input('image_path_5'), debug = True)
result_6 = detect_rubiks_cube_colors(input('image_path_6'), debug = True)
result = [result_1,result_2,result_3,result_4,result_5,result_6]

face_U = face_R = face_F = face_D = face_L = face_B = None

#assigning random faces to order
for char in result:
    if char[1][1] == 'red':
        face_R = char
    elif char[1][1] == 'white':
        face_U = char
    elif char[1][1] == 'green':
        face_F = char
    elif char[1][1] == 'yellow':
        face_D = char
    elif char[1][1] == 'orange':
        face_L = char
    elif char[1][1] == 'blue':
        face_B = char
face = [face_U,face_R,face_F,face_D,face_L,face_B]
if None in [face_U, face_R, face_F, face_D, face_L, face_B]:
    raise ValueError("Could not identify all six faces. Please check image quality or color detection logic.")

# faces: list of 6 faces in order [U, R, F, D, L, B], each face is a 3x3 matrix
def convert_faces_to_kociemba_string(faces):

    color_to_face = {
        "white": "U",
        "red": "R",
        "green": "F",
        "yellow": "D",
        "orange": "L",
        "blue": "B"
    }
    
    face_string = ""
    for char in faces:
        for row in char:
            for color in row:
                face_string += color_to_face.get(color,'X')
    return face_string

face_input = convert_faces_to_kociemba_string(face)
print("Face Input:", face_input)
print("Length:", len(face_input))

from collections import Counter
print("Color Counts:", Counter(face_input))


solution = kociemba.solve(face_input)

with open('kociemba_solution.txt','w') as file:
    file.write(solution)

print(solution)
