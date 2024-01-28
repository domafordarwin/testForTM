import streamlit as st
import cv2

from keras.models import load_model  # TensorFlow is required for Keras to work

st.title(f'Is it {classes[0]} or {classes[1]}!?')
img_file_buffer = st.camera_input(f"Take a picture of {classes[0]} or {classes[1]}")

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(cv2_img, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    probabilities = model.predict(image)

if probabilities[0,0] > 0.8:
        prob = round(probabilities[0,0] * 100,2)
        st.write(f"I'm {prob}% sure that's {classes[0]}!")
    elif probabilities[0,1] > 0.8:
        prob = round(probabilities[0,1] * 100,2)
        st.write(f"I'm {prob}% sure that's {classes[1]}!")
    else:
        st.write("I'm not confident that I know what this is! ")

st.balloons()
