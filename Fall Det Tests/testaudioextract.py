import cv2
import threading

def run_audio_listener(video_capture):
    while True:
        query = listen(video_capture)
        if "help" in query:
            send_telegram_message("Someone needs Help!!!!")

if __name__ == "__main__":
    video_capture = cv2.VideoCapture("/dataset/custom/fall/darkbgfall.mp4")
    #video_capture = cv2.VideoCapture(0)

    video_thread = threading.Thread(target=fall_det, args=(video_capture,))
    video_thread.start()

    audio_thread = threading.Thread(target=run_audio_listener, args=(video_capture,))
    audio_thread.start()