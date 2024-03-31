import cv2

# Load the video
cap = cv2.VideoCapture(r"C:\FilesBeth15022024\ludus project dataset\dataset_warriorSlap_ludus-20240330T195138Z-001\dataset_warriorSlap_ludus\warrior_slap_videos\VID_20240315_154800.mp4")
#2 people video: "C:\FilesBeth15022024\ludus project dataset\dataset_warriorSlap_ludus-20240330T195138Z-001\dataset_warriorSlap_ludus\warrior_slap_videos\Video_2024-03-17_at_09.23.47_338d6222.mp4"

# Get the video properties (frame width, frame height, and frames per second)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

# Load the Haar cascades for upper body and full body detection
upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect upper bodies in the frame
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect full bodies in the frame if no upper bodies are detected
    if len(upper_bodies) == 0:
        full_bodies = full_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in full_bodies:
            cropped_frame = frame[y:y+h, x:x+w]
            out.write(cropped_frame)  # Write the cropped frame to the output video
            # Display the cropped frame
            cv2.imshow('Cropped Frame', cropped_frame)
    else:
        for (x, y, w, h) in upper_bodies:
            cropped_frame = frame[y:y+h, x:x+w]
            out.write(cropped_frame)  # Write the cropped frame to the output video
            # Display the cropped frame
            cv2.imshow('Cropped Frame', cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer, and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()
