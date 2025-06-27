import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk, ImageOps
from tkinter import filedialog
from tkinter import Frame
import requests  # Add this import at the top of your file

# Color ranges (HSV)
color_ranges = [
    ('red',    [0, 70, 120],    [10, 255, 255],   (128,0,128)),
    ('red2',   [170, 70, 70],   [180, 255, 255],  (128,0,128)),
    ('orange', [8, 80, 80],     [20, 255, 255],   (128,0,128)),
    ('yellow', [22, 80, 80],    [35, 255, 255],   (128,0,128)),
    ('green',  [40, 40, 40],    [80, 255, 255],   (128,0,128)),
    ('blue',   [94, 80, 70],    [126, 255, 255],  (128,0,128)),
    ('white', [0, 0, 200], [5, 30, 255], (128,0,128))
]

# Initialize GUI
root = tk.Tk()
root.title("Rubik's Cube Solver")

# Configure the root grid to make components responsive
root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(1, weight=0)
root.grid_rowconfigure(2, weight=0)
root.grid_columnconfigure(0, weight=1)

# Frames to store captured faces
captured_faces = [None] * 6

# Initialize matrix to store detected pieces for Kociemba solver
cube_matrix = {face: [[None for _ in range(3)] for _ in range(3)] for face in ["Front", "Back", "Left", "Right", "Top", "Bottom"]}

# Function to process frame and detect colors, returning detected pieces
def process_frame(frame, image_name="Image", display_masks=False):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    result = frame.copy()
    detected_pieces = []

    for name, lower, upper, box_color in color_ranges:
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))


        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.04*cv2.arcLength(cnt,True), True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and 800 < area < 30000:  # Ignore very large areas (likely background)
                x, y, w, h = cv2.boundingRect(approx)
                if 0.7 < w/h < 1.3:
                    # Check if the contour is near the edges of the image (likely background)
                    if x < 10 or y < 10 or x + w > frame.shape[1] - 10 or y + h > frame.shape[0] - 10:
                        continue  # Skip contours near the edges

                    cv2.rectangle(result, (x, y), (x+w, y+h), box_color, 2)
                    cv2.putText(result, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    cv2.circle(result, (center_x, center_y), 5, box_color, -1)
                    detected_pieces.append((center_x+9*center_y, center_x, name))

    # Sort detected pieces lexically
    detected_pieces = sorted(set(detected_pieces))  # Remove duplicates and sort

    for j in range(len(detected_pieces)):
        detected_pieces[j] = detected_pieces[j][2]  # Keep only the piece names


    return result, detected_pieces

# Modify the update_captured_faces function to accept detected pieces
def update_captured_faces():
    for i, face in enumerate(captured_faces):
        if face is not None:
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(face_rgb)
            img = ImageOps.fit(img, (250, 250))  # Resize to 250x250 for optimal display
            img_tk = ImageTk.PhotoImage(img)
            face_labels[i].config(image=img_tk)
            face_labels[i].image = img_tk

# Update the upload_face function to display the full annotated image in a new window
def upload_face(index):
    global captured_faces, cube_matrix
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", ".png;.jpg;*.jpeg")])
    if file_path:
        uploaded_image = cv2.imread(file_path)
        processed, detected_pieces = process_frame(uploaded_image)
        print(f"Detected pieces: {detected_pieces}")
        captured_faces[index] = processed
        update_captured_faces()
        print(f"Detected pieces for {face_names[index]}: {detected_pieces}")
        face_labels[index].config(text=f"{face_names[index]}\nDetected: {', '.join(detected_pieces)}")

        # Display the full annotated image in a new window
        cv2.imshow(f"Annotated {face_names[index]}", processed)
        cv2.waitKey(0)
        cv2.destroyWindow(f"Annotated {face_names[index]}", processed)

        # Store detected pieces in the matrix
        for i, piece in enumerate(detected_pieces[:9]):  # Ensure only 9 pieces are stored
            row, col = divmod(i, 3)
            cube_matrix[face_names[index]][row][col] = piece

        print(f"Updated matrix for {face_names[index]}: {cube_matrix[face_names[index]]}")

# Function to generate Kociemba string from the cube matrix
def generate_kociemba_string():
    color_to_kociemba = {
        'red': 'R',
        'orange': 'O',
        'yellow': 'Y',
        'green': 'G',
        'blue': 'B',
        'white': 'W'
    }
    face_order = ["Top", "Right", "Front", "Bottom", "Left", "Back"]  # Standard Kociemba face order
    kociemba_string = ''

    for face in face_order:
        for row in cube_matrix[face]:
            for piece in row:
                kociemba_string += color_to_kociemba.get(piece, '?')  # Map color to Kociemba notation or '?' for undefined

    return kociemba_string

# Update the import_images function to fix matrix population and display only the back image
def import_images():
    global cube_matrix
    image_files = {
        "Front": "F.png",
        "Back": "B.png",
        "Left": "L.png",
        "Right": "R.png",
        "Top": "T.png",
        "Bottom": "D.png"
    }

    for face, file_name in image_files.items():
        try:
            uploaded_image = cv2.imread(file_name)
            if uploaded_image is None:
                print(f"Error: {file_name} not found or could not be opened.")
                continue

            processed, detected_pieces = process_frame(uploaded_image)
            if face == "Back":
                processed, detected_pieces = process_frame(uploaded_image, display_masks=True)
            print(f"Detected pieces for {face}: {detected_pieces}")
            captured_faces[face_names.index(face)] = processed

            # Correctly populate the matrix
            matrix_index = 0
            for row in range(3):
                for col in range(3):
                    if matrix_index < len(detected_pieces):
                        cube_matrix[face][row][col] = detected_pieces[matrix_index]
                        matrix_index += 1

            print(f"Updated matrix for {face}: {cube_matrix[face]}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    # Refresh the Tkinter window with updated face images
    update_captured_faces()

    # Generate and display the Kociemba string
    kociemba_string = generate_kociemba_string()
    print(f"Kociemba String: {kociemba_string}")
    kociemba_label.config(text=f"Kociemba String:\n{kociemba_string}")

    # Wait for user to close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to send the solution string to the NodeMCU server
def send_to_nodemcu(solution_string):
    url = "http://localhost:9000/solution"  # Replace <nodemcu-ip-address> with the actual IP
    data = {"solution": solution_string}
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("Solution sent successfully to NodeMCU!")
        else:
            print(f"Failed to send solution. Status code: {response.status_code}")
    except Exception as e:
        print(f"Error sending solution to NodeMCU: {e}")

# Update the solve_cube function to send the solution to NodeMCU
def solve_cube():
    # Generate the Kociemba string from the current cube matrix
    kociemba_string = generate_kociemba_string()
    print(f"Kociemba String for solving: {kociemba_string}")

    # Placeholder for solving logic (replace with actual solver integration)
    solution_moves = "U R U' L F2 D B' R2"  # Example solution moves

    # Display the solution moves in the GUI
    solution_label.config(text=f"Solution Moves:\n{solution_moves}")

    # Send the solution string to the NodeMCU server
    send_to_nodemcu(solution_moves)

# Create a frame for the face labels and buttons
faces_frame = Frame(root)
faces_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

# Configure faces_frame grid for better layout
for i in range(2):
    faces_frame.grid_rowconfigure(i, weight=1)
for i in range(3):
    faces_frame.grid_columnconfigure(i, weight=1)

# Labels for captured faces (remove upload buttons)
face_labels = []
face_names = ["Front", "Back", "Left", "Right", "Top", "Bottom"]

for i, face_name in enumerate(face_names):
    row = i // 3  # Arrange in 2 rows of 3 columns
    col = i % 3
    
    frame = Frame(faces_frame, relief=tk.RAISED, borderwidth=2)
    frame.grid(row=row, column=col, padx=10, pady=10, sticky="nsew")

    lbl = Label(frame, text=face_name, bg="lightgray", font=("Arial", 12, "bold"))
    lbl.pack(pady=10, fill="both", expand=True)
    face_labels.append(lbl)

# Update the UI to include an import button
import_button = Button(root, text="Import Images", command=import_images, width=20, height=2, bg="lightblue")
import_button.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

# Add a label to display the Kociemba string
kociemba_label = Label(root, text="Kociemba String:", width=50, height=5, bg="lightgray", wraplength=400, justify="center")
kociemba_label.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

# Add a button to solve the cube
solve_button = Button(root, text="Solve Cube", command=solve_cube, width=20, height=2, bg="lightgreen")
solve_button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

# Add a label to display the solution moves
solution_label = Label(root, text="Solution Moves:", width=50, height=5, bg="lightgray", wraplength=400, justify="center")
solution_label.grid(row=4, column=0, padx=10, pady=10, sticky="ew")

# Start GUI loop
root.mainloop()
