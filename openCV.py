import cv2
import time
import poseModule as pm

def draw_label(img, text, position, background_color, text_color=(255, 255, 255), font_scale=0.7):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, 2)
    x, y = position
    cv2.rectangle(img, (x, y - text_height - 10), (x + text_width, y + 10), background_color, -1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, 2)

def process_frame(img, detector, angleThreshold):
    img = detector.findPose(img, draw=False)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        angleRight = detector.findAngle(img, 11, 13, 15)
        angleLeft = detector.findAngle(img, 12, 14, 16)

        message = "Angle is acceptable"
        background_color = (0, 128, 0)  
        if not is_angle_acceptable(angleRight, angleThreshold) or not is_angle_acceptable(angleLeft, angleThreshold):
            message = "Angle is too large!"
            background_color = (0, 0, 128) 
        
        draw_label(img, message, (50, 80), background_color)
    
    return img

def is_angle_acceptable(angle, threshold=60):
    return angle <= threshold

def main():
    cap = cv2.VideoCapture("Videos/2022-05-05 11.32.11_Camera 1_6.mov")
    detector = pm.PoseDetector()
    pTime = 0
    angleThreshold = 60

    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  
    cv2.resizeWindow("Image", 1920, 1080)  

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read video frame. Skipping to next frame or ending if at video end.")
            continue

        img = process_frame(img, detector, angleThreshold)

        # Calculate and display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        draw_label(img, f"FPS: {int(fps)}", (70, 20), (10, 10, 10))

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  # Escape key to exit
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
