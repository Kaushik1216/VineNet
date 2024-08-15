from io import BytesIO
import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image
import io
import cv2
import numpy as np
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode
import threading
import asyncio

EXAMPLE_NO = 1

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def streamlit_menu(example=2):
        selected = option_menu(
            menu_title=None,
            options=["image", "mask" , "video" ],
            icons=["image", "mask", "camera-video-fill" ],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
        )
        return selected

selected = streamlit_menu(example=EXAMPLE_NO)

def load_model():
    model_path = "./best.pt"
    model = YOLO(model_path)
    return model

if selected == "image":
    st.title(f"You have selected {selected}")
    st.title("Note: Video (Live) segmentation may not work on the host website due to the limitations of the free hosting service, as it requires high computational power that is not provided in the free version.")
    file = st.file_uploader("Upload file", type=["jpeg", "png", "jpg"])
    show_file = st.empty()
    FRAME_WINDOW = st.image([])

    class ImageClass(object):
        def __init__(self):
            self.model = load_model()
            self.image = np.zeros(6, np.uint8)
    
        def run(self):

            if not file:
                image_path = './image.png'
                image = Image.open(image_path)
                self.image = np.array(image)
                self.process()
                return

            if isinstance(file, BytesIO):
                st.image(file, caption="Uploaded Image")
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))
                self.image = np.array(image)
                self.process()

        def process(self):

                image = self.image
                image = image[:, :, ::-1]

                with torch.no_grad():
                    output = self.model.predict(image)
                    output = cv2.cvtColor(output[0].plot(), cv2.COLOR_BGR2RGB)
                    FRAME_WINDOW.image(output, use_column_width=True, caption="Segmentated Image")
                    is_success, buffer = cv2.imencode(".png", output)
                    if is_success:
                        st.download_button(
                            label="Download Segmented image",
                            data=buffer.tobytes(),
                            file_name="predicted_image.png",
                            mime="image/png"
                        )

    if __name__ ==  "__main__":
        imageobject = ImageClass()
        imageobject.run()

if selected == "video":
    st.title(f"You have selected {selected}")
    FRAME_WINDOW = st.image([])
    RTC_CONFIGURATION = RTCConfiguration(
       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    class VideoClass(object):
        def __init__(self):
            self.model = load_model()
            self.lock = threading.Lock()
            self.img = None

        async def run(self):

            def video_frame_callback(frame):
                img = frame.to_ndarray(format="bgr24")
                with self.lock:
                    self.img = img

                return frame

            ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False},mode=WebRtcMode.SENDRECV,
                  rtc_configuration=RTC_CONFIGURATION,async_processing=True)

            while ctx.state.playing:
                with self.lock:
                    img = self.img
                if img is None:
                    continue

                with torch.no_grad():
                    output = self.model.predict(img)
                    output = output[0].plot()
                FRAME_WINDOW.image(cv2.cvtColor(output, cv2.COLOR_BGR2RGB), use_column_width=True)

        async def cleanup(self):
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            [task.cancel() for task in tasks]
            await asyncio.gather(*tasks, return_exceptions=True)
            asyncio.get_event_loop().stop()

    if __name__ ==  "__main__":
        videoobject = VideoClass()
        try:
            asyncio.run(videoobject.run())
        except KeyboardInterrupt:
            asyncio.run(videoobject.cleanup())


if selected == "mask":
    st.title(f"You have selected {selected}")
    file = st.file_uploader("Upload file", type=["jpeg", "png", "jpg"])
    show_file = st.empty()
    FRAME_WINDOW = st.image([])

    class MaskClass(object):
        def __init__(self):
            self.model = load_model()
            self.image = np.zeros(6, np.uint8)
    
        def run(self):

            if not file:
                image_path = './image.png'
                image = Image.open(image_path)
                self.image = np.array(image)
                self.process()
                return

            if isinstance(file, BytesIO):
                st.image(file, caption="Uploaded Image")
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))
                self.image = np.array(image)
                self.process()

        def process(self):

                image = self.image
                image = image[:, :, ::-1]

                with torch.no_grad():
                    output = self.model.predict(image)
                    for r in output:
                        img = np.copy(r.orig_img)

                        b_mask = np.zeros(img.shape[:2], np.uint8)

                        for ci,c in enumerate(r):
                            #  Get detection class name
                            label = c.names[c.boxes.cls.tolist().pop()]
                            #  Extract contour result
                            contour = c.masks.xy.pop()
                            #  Changing the type
                            contour = contour.astype(np.int32)
                            #  Reshaping
                            contour = contour.reshape(-1, 1, 2)

                            # Draw contour onto mask
                            try:
                                predict_mask = cv2.drawContours(b_mask,
                                                    [contour],
                                                    -1,
                                                    (255, 255, 255),
                                                    cv2.FILLED)
                            except:
                                continue
                    FRAME_WINDOW.image(b_mask, use_column_width=True, caption="Segmentated Mask")
                    is_success, buffer = cv2.imencode(".png", b_mask)
                    st.download_button(
                        label="Download Segmented Mask",
                        data=buffer.tobytes(),
                        file_name="predicted_image.png",
                        mime="image_mask/png"
                    )

    if __name__ ==  "__main__":
        helper = MaskClass()
        helper.run()