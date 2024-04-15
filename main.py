# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import cv2 as cv
import time
import process_key_frame
import feature_process
import camera_motion_ransaclike_v1
import  project_motion_v1
import constant_defination as const
import motion_compensation as mocom
from PIL import Image  
import PIL
import os
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
previous_frame = None


def start_cam_motion_estimation(path):
    global previous_frame

    cap = cv.VideoCapture(path)
    fps_rate = cap.get(cv.CAP_PROP_FPS)
    frame_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    frame_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv.VideoWriter('VideoData/Video2/avi_file/Video2_MJPG_avi_compress.avi', cv.VideoWriter_fourcc(*'MJPG'), int(fps_rate) ,(int(frame_width), int(frame_height)),isColor=True)

    print("Reading ", path, " at ", frame_width, "x", frame_height, " @", fps_rate, "fps.")
    trj_plot = project_motion_v1.Trajectory("Camera Motion")
    frames_considered = 0
    time_snap = time.time()
    fps = 0
    processingTime = time.time()
    while cap.isOpened():
        succ, frame = cap.read()

        if succ:  # Frame was correctly acquired
            frames_considered += 1

            time_elapsed = time.time() - time_snap
            avg_fps = float("{0:.2f}".format(frames_considered / time_elapsed))

            new_frame = process_key_frame.KeyFrame(frame.copy(),
                                          0 if previous_frame is None else previous_frame.get_frame_id() + 1)
            new_frame.find_key_points()

            if previous_frame is not None:
                matched_features = feature_process.match_features(previous_frame, new_frame, False)
                top_good_matches = feature_process.find_top_good_matches(previous_frame, new_frame, matched_features, const.TOP_MATCHES, True)
                if(len(top_good_matches) > 0):
                    del_x, del_y, del_theta = camera_motion_ransaclike_v1.estimate_camera_motion(previous_frame, new_frame, top_good_matches)
                    trj_plot.process_motion_hypothesis(del_x, del_y, del_theta)
                    diff_img=mocom.motion_compensat(previous_frame, new_frame, del_x, del_y, del_theta)
                    # dct with color channel
                    # compressed_frame = compensated_video_compression.dct_compress(diff_img, block_size=8) #dct

                    # Start Resize using PIL

                    # compressed_frame=Image.fromarray(diff_img) 
                    # compressed_frame=compressed_frame.resize((int(frame_width), int(frame_height)),Image.Resampling.LANCZOS)
                    # # compressed_frame=compressed_frame.quantize(256)
                    # compressed_frame=np.array(compressed_frame) 
                    # filename = os.path.join('media/outpy_processed_frame', f"image_{frames_considered}.png")
                    # cv.imwrite(filename, compressed_frame)
                    #Stop Resizing using PIL
                
                    # First, convert 'compressed_frame' to a PIL Image
                    # compressed_frame_pil = Image.fromarray(compressed_frame)
                    # # Dequantize the image
                    # dequantized_frame = compressed_frame_pil.convert("RGB")
                    # # Convert the dequantized frame back to a NumPy array
                    # dequantized_frame_np = np.array(dequantized_frame)

                    # dct with gray color

                    # gray_frame = cv.cvtColor(diff_img, cv.COLOR_BGR2GRAY) 
                    # gray_frame_float32 = np.float32(gray_frame)
                    # compressed_frame = cv.dct(gray_frame_float32)
                    # idct_frame = cv.idct(compressed_frame)
                    # idct_frame_uint8 = np.uint8(idct_frame)
                    # idct_frame_uint8=cv.cvtColor(idct_frame_uint8,cv.COLOR_GRAY2BGR)  #idct
                    out.write(diff_img)
                # time.sleep(0.25)
                t = time.time()
                fps = round(1 / (t - processingTime), 2)
                processingTime = t
            previous_frame = new_frame

            cv.putText(frame, str(fps) + " FPS", (10, 30), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0))
            cv.imshow("Untracked Features", frame)
        else:
            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    # out.release()
    cv.destroyAllWindows()


# Press the green button in the gutter to run the script.


if __name__ == '__main__':
    start_cam_motion_estimation('VideoData/Video2/avi_file/Video2_MJPG_avi.avi')
