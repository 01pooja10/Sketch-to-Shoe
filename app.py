import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
from code.model import Generator
from code import config
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
from torchvision.utils import save_image

st.header('Sketch to Shoe')
st.subheader('Image to image translation: a reimplementation of pix2pix GAN')
st.write('Project Sketch-to-Shoe is a thorough yet comprehensible reimplementation of the pix2pix GAN paper (Isola et al.)')
st.write('It adheres to the PyTorch framework and is trained on various sketches/images of shoes.')
st.image('assets/120_AB.jpg')
stwid = st.sidebar.slider('Width of brush: ',1,10,2)
color = st.sidebar.color_picker('Choose the color you want: ')
img_dat = st_canvas(stroke_width=stwid,stroke_color=color,
                background_color='#000',height=300,width=400,
                drawing_mode='freedraw',key='canvas')
cv2.imwrite(r'samples/inp.jpg',img_dat.image_data)
img = Image.open(r'samples/inp.jpg').resize((256, 256), Image.ANTIALIAS).convert('RGB')

img = np.asarray(img, dtype='uint8')

transf = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
img = transf(img)
img = img.unsqueeze(0)
img = img.to(config.device)
x = st.button('pix2pix')
gen = Generator(in_channels=3).to(config.device)
saved = torch.load(r'D:\pix2pix_models\gen_model.pth',map_location=config.device)
gen.load_state_dict(saved['model'])
gen.eval()
if x:
    with torch.no_grad():
        out = gen(img)
        out = out*0.5 + 0.5
        save_image(out,r'C:\Users\Pooja\Documents\ML_projects\Sketch-to-Shoe\samples\display.png')

    imout = Image.open(r'samples/display.png')
    st.image(imout)
    st.write('There you go!')
    st.subheader('End of demo')
