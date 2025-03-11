import os
import cv2
import numpy as np
import tarfile
import tempfile

def video_grab_cut(video_path: str, frame_num : int) -> np.array(tuple[np.ndarray[any, np.dtype]]):
    """
    :param video_path: file path to the video
    :param frame_num: number of frames to extract (based on initial number of frame)
    :return frames: an array of tuples containing a greyscale image of foreground and background
    """

    capture: cv2.VideoCapture = cv2.VideoCapture(video_path)
    frames = []

    # retrieve initial number of frame

    for frame_index in range(frame_num):

        # get frame at a certain position and read it
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            n, m = frame.shape[:2]

            # filter the frame for easier foreground / background detection
            filtered_frame = cv2.medianBlur(frame, 5)
            filtered_frame = cv2.bilateralFilter(filtered_frame, 9, 20, 75)
            rect = (5, 0, int(m * .9), n)

            # Create mask and background/foreground models
            mask = np.zeros(filtered_frame.shape[:2], np.uint8)
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Apply GrabCut
            cv2.grabCut(filtered_frame, mask, rect, bgd_model, fgd_model, 20, cv2.GC_INIT_WITH_RECT)

            # Modify mask to create binary output
            mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, np.ones((5, 5)))

            # get foreground / background
            foreground = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            background = foreground.copy()
            foreground[mask_binary == 0] = 0
            background[mask_binary == 1] = 0
            frames.append((foreground, background))

            cv2.imshow("result", foreground)
            cv2.waitKey(10)
    return np.array(frames)

def download_frames(main_path, frames, path, percentage):
    foregrounds = frames[:, 0]
    backgrounds = frames[:, 1]
    for i in range(len(percentage)):
        image_path = f"{path.folder}_{path.avi}_{path.folder}_{percentage[i]}"
        the_path = os.path.join(main_path, image_path)

        cv2.imwrite(fr"{the_path}_foreground.png", foregrounds[i])
        cv2.imwrite(fr"{the_path}_background.png", backgrounds[i])

class AviPath:
    def __init__(self,secondary, folder, avi, avi_path):
        self.secondary = secondary
        self.folder =folder
        self.avi = avi
        self.avi_path = avi_path

if __name__ == "__main__":
    primary_file_path = r"D:\DeepfakeTIMIT\DeepfakeTIMIT.tar.gz"

    avi_files_list = []
    frame_num : int = 25

    with tarfile.open(primary_file_path, "r") as tar:
        # List of files within the tar file
        file_names = tar.getnames()

        # Loop over secondary file paths, assuming they are directory names within the tar archive
        secondary_file_path = ["higher_quality", "lower_quality"]
        for secondary in secondary_file_path:
            for name in file_names:
                if secondary in name:

                    # grab only the .avi files
                    if name.endswith(".avi"):

                        #stores the separate directory parts to be used later
                        path_parts = name.split("/")
                        path = AviPath(secondary, path_parts[2], path_parts[3], name)

                        # get temporary files to read from
                        video_file = tar.extractfile(name)
                        video_data = video_file.read()
                        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
                            tmpfile.write(video_data)
                            tmpfile_path = tmpfile.name

                        frames = video_grab_cut(tmpfile_path, frame_num)

                        # download extracted images
                        #download_frames(r"D:\DeepfakeTIMIT\ProcessedImage", frames, path, frame_num)


