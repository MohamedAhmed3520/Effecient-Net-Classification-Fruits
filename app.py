import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import time

# --------------------------------
# Page
# --------------------------------

st.set_page_config(
    page_title="Fruit Classifier",
    page_icon="🍎",
    layout="wide"
)

# --------------------------------
# CSS
# --------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#fff8e7,#fff3cd,#ffffff);
}

#MainMenu,footer,header{
visibility:hidden;
}

.title{
text-align:center;
font-size:48px;
font-weight:bold;
color:#ff6f00;
}

.subtitle{
text-align:center;
color:#555;
margin-bottom:30px;
}

.card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 0px 10px rgba(0,0,0,.12);
}

.stButton>button{
width:100%;
height:50px;
font-size:18px;
font-weight:bold;
background:#ff9800;
color:white;
border-radius:10px;
}

.terminal{
background:#111;
color:#00ff66;
padding:15px;
border-radius:12px;
font-family:monospace;
white-space:pre-wrap;
}

</style>

<div class="title">

🍎 Fruit Classifier

</div>

<div class="subtitle">

EfficientNet-B0 • PyTorch • 100 Fruit Classes

</div>

""",unsafe_allow_html=True)

# --------------------------------
# Device
# --------------------------------

device=torch.device(
"cuda"
if torch.cuda.is_available()
else
"cpu"
)

# --------------------------------
# Model
# --------------------------------

@st.cache_resource
def load():

    model=models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features=model.classifier[1].in_features

    model.classifier=nn.Sequential(

        nn.Dropout(.2),

        nn.Linear(in_features,100)

    )

    model.load_state_dict(
        torch.load(
            "best_model.pth",
            map_location=device
        )
    )

    model.to(device)

    model.eval()

    return model

model=load()

# --------------------------------
# Transform
# --------------------------------

transform=transforms.Compose([

transforms.Resize((224,224)),

transforms.ToTensor(),

transforms.Normalize(

[0.485,0.456,0.406],

[0.229,0.224,0.225]

)

])

# --------------------------------
# Classes
# --------------------------------

class_names=[
"abiu","acai","acerola","ackee","ambarella","apple","apricot","avocado","banana",
"barbadine","barberry","betel_nut","bitter_gourd","black_berry","black_mullberry",
"brazil_nut","camu_camu","cashew","cempedak","chenet","cherimoya","chico","chokeberry",
"cluster_fig","coconut","corn_kernel","cranberry","cupuaçu","custard_apple","damson",
"dewberry","dragonfruit","durian","eggplant","elderberry","emblic","feijoa","fig",
"finger_lime","gooseberry","goumi","grape","grapefruit","greengage","grenadilla","guava",
"hard_kiwi","hawthorn","hog_plum","horned_melon","indian_strawberry","jaboticaba",
"jackfruit","jalapeno","jamaica_cherry","jambul","jocote","jujube","kaffir_lime",
"kumquat","lablab","langsat","longan","mabolo","malay_apple","mandarine","mango",
"mangosteen","medlar","mock_strawberry","morinda","mountain_soursop","oil_palm","olive",
"otahiete_apple","papaya","passion_fruit","pawpaw","pea","pineapple","plumcot",
"pomegranate","prickly_pear","quince","rambutan","raspberry","redcurrant","rose_hip",
"rose_leaf_bramble","salak","santol","sapodilla","sea_buckthorn","strawberry_guava",
"sugar_apple","taxus_baccata","ugli_fruit","white_currant","yali_pear","yellow_plum"
]

# --------------------------------
# Layout
# --------------------------------

left,right=st.columns([2,1])

with left:

    uploaded=st.file_uploader(
        "Upload a fruit image",
        type=["jpg","png","jpeg"]
    )

    if uploaded:

        img=Image.open(uploaded).convert("RGB")

        st.image(img,use_container_width=True)

with right:

    st.markdown("""

<div class="card">

<h3>🤖 Model Information</h3>

<b>Architecture</b><br>

EfficientNet-B0

<br><br>

<b>Framework</b><br>

PyTorch

<br><br>

<b>Dataset</b><br>

<a href="https://www.kaggle.com/datasets/icebearogo/fruit-classification-dataset" target="_blank">
📂 Fruit Classification (Kaggle)
</a>

<b>Classes</b><br>

100 Fruits

</div>

""",unsafe_allow_html=True)

# --------------------------------
# Prediction
# --------------------------------

if uploaded and st.button("🚀 Predict"):

    progress=st.progress(0)

    for i in range(100):

        progress.progress(i+1)

        time.sleep(.005)

    progress.empty()

    tensor=transform(img).unsqueeze(0).to(device)

    with torch.no_grad():

        out=model(tensor)

        probs=torch.softmax(out,1)[0]

    values,indices=torch.topk(probs,5)

    terminal=st.empty()

    terminal.markdown(f"""

<div class='terminal'>

> MODEL LOADED

> IMAGE PREPROCESSED

> RUNNING EfficientNet-B0

> RESULT

{class_names[indices[0]]}

</div>

""",unsafe_allow_html=True)

    c1,c2=st.columns(2)

    with c1:

        st.metric(

            "Prediction",

            class_names[indices[0]],

            f"{values[0]*100:.2f}%"

        )

        df=pd.DataFrame({

            "Fruit":[class_names[i] for i in indices],

            "Confidence (%)":[float(v)*100 for v in values]

        })

        st.dataframe(

            df,

            hide_index=True,

            use_container_width=True

        )

    with c2:

        fig,ax=plt.subplots(figsize=(6,3))

        ax.barh(

            [class_names[i] for i in indices][::-1],

            [float(v)*100 for v in values][::-1]

        )

        ax.set_xlabel("Confidence (%)")

        st.pyplot(fig)

st.markdown("---")

st.markdown("""

<center>

🍎 EfficientNet-B0 • PyTorch • Streamlit

</center>

""",unsafe_allow_html=True)
