import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from keras_retinanet import models
from keras_retinanet.utils.image import preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import streamlit as st



model = models.load_model('resnet.h5')
class_names = [l.rstrip() for l in open('coco_categories.txt')]

def predictions(img, threshold=0.6):
  im = np.array(img)
  print("im.shape:", im.shape)

  # if there's a PNG it will have alpha channel
  im = im[:,:,:3]
  
  ### plot predictions ###

  # get predictions
  imp = preprocess_image(im)
  imp, scale = resize_image(im)

  boxes, scores, labels = model.predict_on_batch(
    np.expand_dims(imp, axis=0)
  )

  # standardize box coordinates
  boxes /= scale

  # loop through each prediction for the input image
  for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can quit as soon
    # as we see a score below threshold
    if score < threshold:
      break

    box = box.astype(np.int32)
    color = label_color(label)
    draw_box(im, box, color=color)

    class_name = class_names[label]
    caption = f"{class_name} {score:.3f}"
    draw_caption(im, box, caption)

  return im
	


st.title("Object Detection")

html_temp = """
<body style="background-color:red;">
<div style="background-color:teal ;padding:10px">
<h2 style="color:white;text-align:center;">Object Detection WebApp</h2>
</div>
</body>
"""
st.markdown(html_temp, unsafe_allow_html=True)

image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
if image_file is not None:
	img = Image.open(image_file)
	st.text("Original Image")
	st.image(img)

if st.button("Compute"):
	result_img= predictions(img)
	st.image(result_img)
	