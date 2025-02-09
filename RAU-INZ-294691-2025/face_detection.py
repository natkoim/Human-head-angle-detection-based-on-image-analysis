import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe.python.solutions.face_mesh as mp_face_mesh
import mediapipe.python.solutions.drawing_utils as mp_drawing
import os
import glob

# Function to save face detection results to a file
def save_data_to_file(data, file_name="output.txt"):
    with open(file_name, 'w') as file:
        for entry in data:
            file.write(f"""
                        Image: {entry.get("image", "N/A")}
                        Face Recognized: {'YES' if entry.get('recognized') else 'NO'}
                       """)
            if entry.get('recognized'):
                file.write(f"Angles (Horizontal, Vertical): {entry.get('angle_horizontal', 0):.2f}, {entry.get('angle_vertical', 0):.2f}\n")
            file.write("\n")
    print(f"Data saved to {file_name}")

# face_data = [{"image": "image_path", "recognized": bool, "angle_horizontal": float, "angle_vertical": float}]

# Function to calculate the horizontal and vertical angles of the face
def calculate_face_angle(landmarks):
    left_eye = np.array([landmarks[33].x, landmarks[33].y])
    right_eye = np.array([landmarks[263].x, landmarks[263].y])
    nose = np.array([landmarks[1].x, landmarks[1].y])

    horizontal_vector = right_eye - left_eye
    vertical_vector = nose - (left_eye + right_eye) / 2

    angle_horizontal = np.degrees(np.arctan2(horizontal_vector[1], horizontal_vector[0]))
    angle_vertical = np.degrees(np.arctan2(vertical_vector[1], vertical_vector[0]))

    return angle_horizontal, angle_vertical

# Function to process and analyze 2D images
def process_2d_picture(method, database_mode, color_mode):
    face_data = []
    if database_mode:
        folder_path = input("Enter the folder path containing the images (.jpg, .png): ")
        if not os.path.exists(folder_path):
            print("Invalid folder path.")
            return
        images = glob.iglob(pathname=glob.escape(f"{folder_path}\\**"), recursive=True)
    else:
        image_path = input("Enter the path of the single image (.jpg, .png): ")
        images = [image_path]

    print("Using Mediapipe Face Mesh for detection and angle estimation...")
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=4, refine_landmarks=True) as face_mesh:
        for image_path in images:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image {image_path}")
                continue

            # Convert to grayscale if monochromatic mode is selected
            if color_mode == 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                    angle_horizontal, angle_vertical = calculate_face_angle(face_landmarks.landmark)
                    print(f"Image: {image_path} | Face Recognized: YES | Angles (H, V): {angle_horizontal:.2f}, {angle_vertical:.2f}")
                    face_data.append({"image": image_path, "recognized": True, "angle_horizontal": angle_horizontal, "angle_vertical": angle_vertical})
            else:
                print(f"Image: {image_path} | Face Recognized: No")
                face_data.append({"image": image_path, "recognized": False})
            cv2.imshow("Image", image)
            cv2.waitKey(0)

    save_data_to_file(face_data)
    cv2.destroyAllWindows()

# Function to process and analyze real-time camera feed
def process_real_time(color_mode):
    cap = cv2.VideoCapture(0)

    print("Using Mediapipe Face Mesh for detection and angle estimation...")
    with mp_face_mesh.FaceMesh(max_num_faces=4, refine_landmarks=True) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Convert to grayscale if monochromatic mode is selected
            if color_mode == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                for i, face_landmarks in enumerate(results.multi_face_landmarks):
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
                    angle_horizontal, angle_vertical = calculate_face_angle(face_landmarks.landmark)
                    print(f"Face Recognized: YES | Angles (H, V): {angle_horizontal:.2f}, {angle_vertical:.2f}")
                    cv2.putText(frame, f"Face recognized: YES | H: {angle_horizontal:.2f}, V: {angle_vertical:.2f}", (10, 30 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Face recognized: NO", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
                print("Face Recognized: No")

            cv2.imshow("The Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# Main function to provide user options
def main():
    print("Head Angle Detection Program")
    print("1. 2D Picture Analysis")
    print("2. Real-Time Camera")
    choice = int(input("Choose an option: "))

    print("1. RGB Mode")
    print("2. Monochromatic Mode")
    color_mode = int(input("Choose color mode: "))

    if choice == 1:
        print("1. Use a database of images (.jpg, .png)")
        print("2. Use a single image (.jpg, .png)")
        database_mode = int(input("Choose image input mode: ")) == 1

        process_2d_picture(method=2, database_mode=database_mode, color_mode=color_mode)
    elif choice == 2:
        process_real_time(color_mode=color_mode)
    else:
        print("Invalid choice.")

if __name__ == "__main__":
    main()
