import cv2
import numpy as np
from PIL import Image  
import PIL

input_file = 'VideoData/Video2/mp4_file/video2.mp4'
cap = cv2.VideoCapture(input_file)

output_file = 'VideoData/Video2/avi_file/Video2_MJPG_avi.avi'
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Change this to your desired codec
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size, isColor=True)

while cap.isOpened():
    ret, frame = cap.read()

    if ret:

        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # gray_frame_float32 = np.float32(gray_frame)
        # compressed_frame = cv2.dct(gray_frame_float32)
        # idct_frame = cv2.idct(compressed_frame)
        # idct_frame_uint8 = np.uint8(idct_frame)
        # idct_frame_uint8 = cv2.cvtColor(idct_frame_uint8, cv2.COLOR_GRAY2BGR)  # Convert back to BGR
         # Write the IDCT frame to the output video file

        # compressed_frame=Image.fromarray(frame) # quantize start
        # compressed_frame=compressed_frame.resize((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))),Image.Resampling.LANCZOS)
        # compressed_frame=np.array(compressed_frame) 
        out.write(frame) 
        # # Display frames
        # cv2.imshow('Original Frame', frame)
        cv2.imshow('DCT Frame', frame)
        # cv2.imshow('IDCT Frame', idct_frame_uint8)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
  