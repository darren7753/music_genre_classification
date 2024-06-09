import os
import subprocess
import utils
import streamlit as st
import torch
import torch.nn as nn
import model.net as net
import model.data_loader as data_loader
import numpy as np
import pandas as pd
import altair as alt
import pickle

from hydralit_components import HyLoader, Loaders
from tempfile import NamedTemporaryFile, TemporaryDirectory
from PIL import Image

st.set_page_config(
    page_title="Music Genre Classification",
    layout="wide",
)

torch.manual_seed(230)

@st.cache_data
def load_pytorch_model_and_params():
    params = utils.Params("experiments/Model_5/params.json")
    model = net.Net(params)
    utils.load_checkpoint(os.path.join("experiments/Model_5", "best" + ".pth.tar"), model, device=torch.device("cpu"))
    return params, model

@st.cache_data
def load_xgboost_model(model_type):
    if model_type == "XGBoost":
        # with open("experiments/XGBoost/base_model_80_20.pkl", "rb") as f:
        #     model = pickle.load(f)
        model = pd.read_pickle("experiments/XGBoost/base_model_80_20.pkl")
    else:
        # with open("experiments/XGBoost/auto_model_80_20.pkl", "rb") as f:
        #     model = pickle.load(f)
        model = pd.read_pickle("experiments/XGBoost/auto_model_80_20.pkl")
    return model

class ConvFeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(ConvFeatureExtractor, self).__init__()
        self.features = nn.Sequential(
            original_model.conv1,
            original_model.bn1,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            original_model.conv2,
            original_model.bn2,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            original_model.conv3,
            original_model.bn3,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            original_model.conv4,
            original_model.bn4,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
            original_model.conv5,
            original_model.bn5,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

def preprocess_track_for_classification(uploaded_file, tmp_test_folder, slice_output_dir):
    # Check if uploaded_file is a file object or bytes data
    if hasattr(uploaded_file, 'read'):  # Check if it's a file object
        # Step 1: Save file to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".mp3", dir=tmp_test_folder) as tmp_file:
            tmp_file.write(uploaded_file.read())
            fname = tmp_file.name
    else:  # It's bytes data
        # Step 1: Save bytes to a temporary location
        with NamedTemporaryFile(delete=False, suffix=".mp3", dir=tmp_test_folder) as tmp_file:
            tmp_file.write(uploaded_file)
            fname = tmp_file.name

    # Step 2: Convert to .wav if necessary
    if fname.endswith(".mp3"):
        dst = os.path.join(tmp_test_folder, os.path.basename(fname).split(".mp")[0] + ".wav")
        command = ['ffmpeg', '-v', '0', '-i', fname, dst]
        subprocess.call(command)
        fname = dst
    else:
        fname = uploaded_file.name

    # Step 3: Create a spectrogram using sox
    dest = os.path.join(tmp_test_folder, os.path.basename(fname).split(".wa")[0] + ".png")
    command = ['sox', fname, '-n', 'remix', '1', 'spectrogram', '-Y', '200', '-X', '50', '-m', '-r', '-o', dest]
    subprocess.call(command)

    if not os.path.exists(dest):
        st.error("Spectrogram creation failed.")
        return None

    # Step 4: Slice and save spectrograms
    os.makedirs(slice_output_dir, exist_ok=True)
    slice_output_dir = os.path.join(slice_output_dir, 'tmp_specs')
    os.makedirs(slice_output_dir, exist_ok=True)  # Ensure that slice_output_dir is created
    new_dest = os.path.basename(dest)
    slice_spec(new_dest, 128, tmp_test_folder, slice_output_dir)

    # Check if slices were created
    if not os.path.exists(slice_output_dir):
        st.error("Slice directory does not exist after creation.")
        return None

    if len(os.listdir(slice_output_dir)) == 0:
        st.error("No slices were generated.")
        return None

    # Return the path of the directory containing the sliced spectrograms
    return slice_output_dir

def slice_spec(fname, size, input_dir, output_dir):
    try:
        file_path = os.path.join(input_dir, fname)
        im = Image.open(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return

    w, h = im.size
    num_slices = int(w / size)

    for i in range(num_slices):
        tmp_im = im.crop((i * size, 0, i * size + size, size))
        slice_filename = f"{fname.split('.')[0]}_{i}.png"
        tmp_im.save(os.path.join(output_dir, slice_filename))

def predict_with_xgboost(xgb_model, features_array):
    genre_mapping = {
        0: "Electronic",
        1: "Experimental",
        2: "Folk",
        3: "Hip-Hop",
        4: "Instrumental",
        5: "International",
        6: "Pop",
        7: "Rock"
    }

    slice_predictions = pd.DataFrame(xgb_model.predict_proba(features_array))
    slice_predictions.columns = [genre_mapping[col] for col in slice_predictions.columns]
    slice_predictions["genre"] = xgb_model.predict(features_array)
    slice_predictions["genre"] = slice_predictions["genre"].map(genre_mapping)
    return slice_predictions

# Streamlit sidebar for file upload
with st.sidebar:
    st.markdown("### Upload an audio file (MP3)", unsafe_allow_html=True)
    audio = st.file_uploader(
        "label",
        type=["mp3"],
        label_visibility="collapsed"
    )

    use_sample = st.checkbox("Use a sample song (AC/DC - Highway to Hell)")

    if use_sample:
        sample_audio_path = "sample_song/AC-DC - Highway to Hell.mp3"
        with open(sample_audio_path, "rb") as f:
            audio = f.read()
        st.audio(audio, format="audio/mp3")
    elif audio:
        st.audio(audio, format="audio/mp3")

    st.markdown("### Choose a model", unsafe_allow_html=True)
    xgb_model_options = st.selectbox(
        "Label",
        options=["XGBoost", "FLAML-Optimized XGBoost"],
        index=None,
        placeholder="Click here to choose a model",
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
        <div style='text-align: center;'>
            <p>Developed with ❤️ by<br>Mathew Darren K. (24050120130042)</p>
        </div>
        
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .social-icons {
                text-align: center;
            }
            .social-icons a {
                color: #FAFAFA;
                font-size: 30px;
                margin: 0 5px;
            }
        </style>
        <div class="social-icons">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.5.0/font/bootstrap-icons.css" />
            <a href='https://www.linkedin.com/in/mathewdarren/' target='_blank'><i class="bi bi-linkedin"></i></a>
            <a href='https://github.com/darren7753' target='_blank'><i class="bi bi-github"></i></a>
            <a href='https://www.instagram.com/darren_matthew_/' target='_blank'><i class="bi bi-instagram"></i></a>
        </div>
    """, unsafe_allow_html=True)

# Main app content
st.html("""
    <style>
        [alt=Logo] {
            height: 3rem;
            border-radius: 10px;
            background-color: #FAFAFA;
            padding: 5px;
        }
    </style>
""")
st.logo(image="S1-Statistika.png", link="https://stat.fsm.undip.ac.id/v1/", icon_image=None)

st.markdown("""
    <h1 style='text-align: center;'>
        Music Genre Classification
    </h1>
""", unsafe_allow_html=True)

st.write("")
st.write("")

if (audio and xgb_model_options) or (use_sample and xgb_model_options):
    with HyLoader("", loader_name=Loaders.pulse_bars):
        with TemporaryDirectory() as tmp_test_folder, TemporaryDirectory() as slice_output_dir:
            params, model = load_pytorch_model_and_params()
            xgb_model = load_xgboost_model(xgb_model_options)
            feature_extractor = ConvFeatureExtractor(model)
            model.to(torch.device("cpu"))
            output_directory = preprocess_track_for_classification(audio, tmp_test_folder, slice_output_dir)

            if output_directory:
                directory, folder_name = os.path.split(output_directory)

                dataloaders = data_loader.fetch_dataloader(["tmp"], directory, params)
                dl = dataloaders["tmp"]

                features_list = []
                for data_batch in dl:
                    with torch.no_grad():
                        features = feature_extractor(data_batch)

                    features_np = features.cpu().numpy()
                    features_list.append(features_np)
                features_array = np.concatenate(features_list, axis=0)

                slice_predictions = predict_with_xgboost(xgb_model, features_array)

                genre_counts = slice_predictions["genre"].value_counts(normalize=True)
                genre_counts_df = pd.DataFrame({"Genre": genre_counts.index, "Percentage": genre_counts.values})

                selector = alt.selection_point(encodings=["x", "y"])

                bar_chart = alt.Chart(genre_counts_df).mark_bar(
                    cornerRadiusTopRight=8,
                    cornerRadiusBottomRight=8
                ).encode(
                    x=alt.X("Percentage:Q", title="", axis=alt.Axis(format="%")),
                    y=alt.Y("Genre:N", title="", sort="-x"),
                    color=alt.condition(selector, "Percentage:Q", alt.value("lightgray"), legend=None),
                    tooltip=["Genre", alt.Tooltip("Percentage", format=".2%")]
                ).add_selection(
                    selector
                ).properties(
                    title="Genre Confidence"
                ).configure_axis(
                    labelFontSize=17,
                    grid=False
                ).configure_title(
                    fontSize=20
                )

                final_prediction = slice_predictions["genre"].mode()[0]
                audio_name = getattr(audio, "name", "AC-DC - Highway to Hell.mp3")
                st.markdown(f"""
                    <h3 style='font-weight: normal;'>
                        The final predicted genre for <b>{audio_name}</b> is <b>{final_prediction}</b>.
                    </h3>
                """, unsafe_allow_html=True)

                with st.container(border=True):
                    st.altair_chart(bar_chart, use_container_width=True)
            
            else:
                st.error("Failed to process the audio file. Please try again.")

else:
    st.info("""
        How it works:
        1. Upload an audio file in MP3 format.
        2. Choose the model for classification: XGBoost or FLAML-Optimized XGBoost.
        3. The audio will be processed into spectrogram slices.
        4. Each slice is individually classified to determine the genre.
        5. The final genre prediction is based on voting from all slices.
    """, icon="ℹ️")