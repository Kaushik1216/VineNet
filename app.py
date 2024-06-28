
 
from enum import Enum
from io import BytesIO, StringIO
from typing import Union
 
import pandas as pd
import streamlit as st
import os
from ultralytics import YOLO
import torch
from torchvision import transforms
from PIL import Image
import io
import time
import cv2
import numpy as np
from pathlib import Path
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer , RTCConfiguration, VideoTransformerBase
import av
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

if selected == "image":
    st.title(f"You have selected {selected}")
    
    
    class FileUpload(object):
        def __init__(self , model):
            self.fileTypes = ["csv", "png", "jpg"]
            self.model = model
    
        def run(self):
            """
            Upload File on Streamlit Code
            :return:
            """

            file = st.file_uploader("Upload file", type=self.fileTypes)
            show_file = st.empty()
            if not file:
                image_path = './image.png'
                image = Image.open(image_path)
                image = np.array(image)
                process(image , model)
                st.image('out.png', caption="Example Segmentated Image")
                show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))
                return

            if isinstance(file, BytesIO):
                st.image(file, caption="Uploaded Image")
                image_bytes = file.read()
                # Convert bytes to PIL Image
                image = Image.open(io.BytesIO(image_bytes))
                image = np.array(image)
                process(image , self.model)
                st.image('out.png', caption="Predicted Image")
            file.close()


    def load_model():
            model_path = "best1.pt"
            model = YOLO(model_path)

            return model
    def process(image , model):

        # Reorder channels (assuming RGB order in PIL image)
            image = image[:, :, ::-1]

            # Perform inference
            with torch.no_grad():
                output = model.predict(image)
                output = output[0]
                output.save("./out.png")

            image_path = './out.png'

            image = Image.open(image_path)

            # st.image(image, caption='vinenetimage')

            with open(image_path, "rb") as file:
                btn = st.download_button(
                        label="Download Seg image",
                        type = 'primary',
                        data=file,
                        file_name="image_k.png",
                        mime="image/png"
                    )

    if __name__ ==  "__main__":
        model = load_model()
        helper = FileUpload(model)
        helper.run()

if selected == "video":
    st.title(f"You have selected {selected}")

    FRAME_WINDOW = st.image([]) #frame window


    st.markdown("""<h2 style='text-align: center;'>Please Check start to start video.</h2>""", unsafe_allow_html=True) #title

    # run = st.checkbox("Start") #checkbox
    #capture video
    model_path = "best1.pt"
    model = YOLO(model_path)# weights are stored and accessed using the path given 
    
    # if run == True: # frame will render 
    #     capture_vid=cv2.VideoCapture(0) # capturing the video 
    #     capture_vid.set(3,640)
    #     capture_vid.set(4,480)
    #     while True:
    #         ret,frame = capture_vid.read()
    #         # frame= cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #convert bgr to rgb format 
    #         results=model.predict(frame) # fitting the model 
    #         result = results[0]

    #         result.save("./out.png")
    #         image_path = './out.png'

    #         image = Image.open(image_path) 
    #         FRAME_WINDOW.image(image) # adding the image in frame window after complete processing
    #         time.sleep(1)

    def callback(frame):
        # img = frame.to_ndarray(format="bgr24")

        # img = cv2.Canny(img , 100, 200)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # return av.VideoFrame.from_ndarray(img , format="bgr24")
        model_path = "best1.pt"
        model = YOLO(model_path)
        results=model.predict(frame) # fitting the model 
        result = results[0]

        result.save("./out.png")
        image_path = './out.png'

        image = Image.open(image_path) 

        return image

    
    webrtc_streamer(key="sample" , video_frame_callback=callable,media_stream_constraints={"video": True, "audio": False})


if selected == "mask":
    st.title(f"You have selected {selected}")

    class FileUpload(object):
        def __init__(self , model):
            self.fileTypes = ["csv", "png", "jpg"]
            self.model = model
    
        def run(self):
            """
            Upload File on Streamlit Code
            :return:
            """

            file = st.file_uploader("Upload file", type=self.fileTypes)
            show_file = st.empty()
            if not file:
                image_path = './image.png'
                image = Image.open(image_path)
                image = np.array(image)
                process(image , model)
                st.image('out.png', caption="Example Segmentated Image")
                show_file.info("Please upload a file of type: " + ", ".join(["csv", "png", "jpg"]))
                return

            if isinstance(file, BytesIO):
                st.image(file, caption="Uploaded Image")
                process(file , self.model)
                st.image('out.png', caption="Predicted Image")
            file.close()

    def load_model():
            model_path = "best1.pt"
            model = YOLO(model_path)

            return model
    def process(image , model):

        # Reorder channels (assuming RGB order in PIL image)
            image = image[:, :, ::-1]

            # Perform inference
            with torch.no_grad():
                output = model.predict(image)
                for r in output:
                    img = np.copy(r.orig_img)
                    img_name = Path(r.path).stem # source image base-name

                    # Create binary mask
                    b_mask = np.zeros(img.shape[:2], np.uint8)

                    # Iterate each object contour (multiple detections)
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
                cv2.imwrite('./out.png', b_mask)

            image_path = './out.png'

            image = Image.open(image_path)

            # st.image(image, caption='vinenetimage')

            with open(image_path, "rb") as file:
                btn = st.download_button(
                        label="Download image",
                        type = 'primary',
                        data=file,
                        file_name="image_k.png",
                        mime="image/png"
                    )

    if __name__ ==  "__main__":
        model = load_model()
        helper = FileUpload(model)
        helper.run()
