import cv2

# Use index 1 for the external webcam
# Removed CAP_DSHOW to use the default system backend
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam index 1.")
    exit()

# Force Auto-Exposure ON (0.75 or 1 usually enables auto)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) 

print("Camera opened. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Empty frame received.")
        break

    cv2.imshow("External Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()