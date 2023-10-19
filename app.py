from PIL import Image
import tensorflow as tf
import streamlit as st
from deep_learning_pipeline import PredictionPipeline

st.title('Malaria Indefected Cell Detection using X-ray Images')
st.write('This Project is built using CNN (Convolutional Neural Networks) Transfer Learning model that helps to predict whether the given X-ray image of the cell is Malaria Infected or Healthy!!')

st.write('')
st.write('')


uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Process the uploaded image here
    with st.container():
        col1, col2 = st.columns([3, 2])
        col1.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        if st.button('Predict!!'):
            pipeline = PredictionPipeline()
            resnet152v2_y_pred, resnet152v2_y_probs, inception_resnetv2_y_preds, inception_resnetv2_y_probs = pipeline.predict(input_img=uploaded_file)
            col2.balloons()
            if resnet152v2_y_pred[0][0] == 1:
                col2.subheader('ResNET 152V2 model: ')
                col2.success(f'{pipeline.CLASS_NAMES[1]}')
                r_acc = '{:.2f}'.format(100*(resnet152v2_y_probs[0][0]))
                col2.success(f'Accuracy: {r_acc}%')
            elif resnet152v2_y_pred[0][0] == 0:
                col2.subheader('ResNET 152V2 model: ')
                col2.success(f'{pipeline.CLASS_NAMES[0]}')
                r_acc = '{:.2f}'.format(100*(1-resnet152v2_y_probs[0][0]))
                col2.success(f'Accuracy: {r_acc}%')
            
            if inception_resnetv2_y_preds[0][0] == 1:
                col2.subheader('Inception Resnet V2 model (Recommended to use): ')
                col2.success(f'{pipeline.CLASS_NAMES[1]}')
                i_acc = '{:.2f}'.format(100*(inception_resnetv2_y_probs[0][0]))
                col2.success(f'Accuracy: {i_acc}%')
            elif inception_resnetv2_y_preds[0][0] == 0:
                col2.subheader('Inception Resnet V2 model: ')
                col2.success(f'{pipeline.CLASS_NAMES[0]}')
                i_acc = '{:.2f}'.format(100*(1-inception_resnetv2_y_probs[0][0]))
                col2.success(f'Accuracy: {i_acc}%')

            elif resnet152v2_y_pred[[0]] == -1:
                col2.error('Error!! Model needs shape (224, 224, 3), but your image is of shape (224, 224,4)')