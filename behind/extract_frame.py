import cv2

cap = cv2.VideoCapture('../video/behind/20260123_145635.mp4')
ret, frame = cap.read()
if ret:
    cv2.imwrite('frame0000.jpg', frame)
    print('Frame extracted to frame0000.jpg')
else:
    print('Failed to extract frame')
cap.release()