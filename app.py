from PIL import Image
import re
import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from pathlib import Path
from beakspeak.data import preprocess_uploaded_image
from beakspeak.params import DATA_DIR

CONFIDENCE_THRESHOLD = 0.53


# --- Bird info helper functions ---
def build_bird_url(name):
    return (
        name
        .replace("'", "")   # remove apostrophes
        .replace(" ", "_")  # spaces → underscores
    )

URL_EXCEPTIONS = {
    "American Three Toed Woodpecker": "American_Three-toed_Woodpecker",
    "Anna Hummingbird": "Annas_Hummingbird",
    "Artic Tern": "Arctic_Tern",
    "Baird Sparrow": "Bairds_Sparrow",
    "Bay Breasted Warbler": "Bay-breasted_Warbler",
    "Bewick Wren": "Bewicks_Wren",
    "Black And White Warbler": "Black-and-white_Warbler",
    "Black Billed Cuckoo": "Black-billed_Cuckoo",
    "Black Capped Vireo": "Black-capped_Vireo",
    "Black Footed Albatross": "Black-footed_Albatross",
    "Black Throated Blue Warbler": "Black-throated_Blue_Warbler",
    "Black Throated Sparrow": "Black-throated_Sparrow",
    "Blue Headed Vireo": "Blue-headed_Vireo",
    "Blue Winged Warbler": "Blue-winged_Warbler",
    "Boat Tailed Grackle": "Boat-tailed_Grackle",
    "Brandt Cormorant": "Brandt's_Cormorant",
    "Brewer Blackbird": "Brewer's_Blackbird",
    "Brewer Sparrow": "Brewer's_Sparrow",
    "Chestnut Sided Warbler": "Chestnut-sided_Warbler",
    "Chuck Will Widow": "Chuck-wills-widow",
    "Clark Nutcracker": "Clarks_Nutcracker",
    "Clay Colored Sparrow": "Clay-colored_Sparrow",
    "Dark Eyed Junco": "Dark-eyed_Junco",
    "Forsters Tern": "Forsters_Tern",
    "Geococcyx": "Greater_Roadrunner",
    "Glaucous Winged Gull": "Glaucous-winged_Gull",
    "Golden Winged Warbler": "Golden-winged_Warbler",
    "Gray Crowned Rosy Finch": "Gray-crowned_Rosy-Finch",
    "Green Tailed Towhee": "Green-tailed_Towhee",
    "Groove Billed Ani": "Groove-billed_Ani",
    "Harris Sparrow": "Harriss_Sparrow",
    "Heermann Gull": "Heermanns_Gull",
    "Henslow Sparrow": "Henslows_Sparrow",
    "Le Conte Sparrow": "LeContes_Sparrow",
    "Long Tailed Jaeger": "Long-tailed_Jaeger",
    "Nelson Sharp Tailed Sparrow": "Nelsons_Sparrow",
    "Olive Sided Flycatcher": "Olive-sided_Flycatcher",
    "Orange Crowned Warbler": "Orange-crowned_Warbler",
    "Pied Billed Grebe": "Pied-billed_Grebe",
    "Red Bellied Woodpecker": "Red-bellied_Woodpecker",
    "Red Breasted Merganser": "Red-breasted_Merganser",
    "Red Cockaded Woodpecker": "Red-cockaded_Woodpecker",
    "Red Eyed Vireo": "Red-eyed_Vireo",
    "Red Faced Cormorant": "Red-faced_Cormorant",
    "Red Headed Woodpecker": "Red-headed_Woodpecker",
    "Red Legged Kittiwake": "Red-legged_Kittiwake",
    "Red Winged Blackbird": "Red-winged_Blackbird",
    "Ring Billed Gull": "Ring-billed_Gull",
    "Rose Breasted Grosbeak": "Rose-breasted_Grosbeak",
    "Ruby Throated Hummingbird": "Ruby-throated_Hummingbird",
    "Sayornis": "Says_Phoebe",
    "Scissor Tailed Flycatcher": "Scissor-tailed_Flycatcher",
    "Scott Oriole": "Scotts_Oriole",
    "Slaty Backed Gull": "Slaty-backed_Gull",
    "Swainson Warbler": "Swainsons_Warbler",
    "White Breasted Kingfisher": "White-breasted_Kingfisher",
    "White Breasted Nuthatch": "White-breasted_Nuthatch",
    "White Crowned Sparrow": "White-crowned_Sparrow",
    "White Eyed Vireo": "White-eyed_Vireo",
    "White Necked Raven": "White-necked_Raven",
    "White Throated Sparrow": "White-throated_Sparrow",
    "Wilson Warbler": "Wilsons_Warbler",
    "Worm Eating Warbler": "Worm-eating_Warbler",
    "Yellow Bellied Flycatcher": "Yellow-bellied_Flycatcher",
    "Yellow Billed Cuckoo": "Yellow-billed_Cuckoo",
    "Yellow Breasted Chat": "Yellow-breasted Chat",
    "Yellow Headed Blackbird": "Yellow-headed Blackbird",
    "Yellow Throated Vireo": "Yellow-throated Vireo"
}

def get_bird_slug(name):
    return URL_EXCEPTIONS.get(name, build_bird_url(name))

DISPLAY_NAME_EXCEPTIONS = {
    "American Three Toed Woodpecker": "American Three-toed Woodpecker",
    "Anna Hummingbird": "Anna's Hummingbird",
    "Artic Tern": "Arctic Tern",
    "Baird Sparrow": "Baird's Sparrow",
    "Bay Breasted Warbler": "Bay-breasted Warbler",
    "Bewick Wren": "Bewick's Wren",
    "Black And White Warbler": "Black-and-white Warbler",
    "Black Billed Cuckoo": "Black-billed Cuckoo",
    "Black Capped Vireo": "Black-capped Vireo",
    "Black Footed Albatross": "Black-footed Albatross",
    "Black Throated Blue Warbler": "Black-throated Blue Warbler",
    "Black Throated Sparrow": "Black-throated Sparrow",
    "Blue Headed Vireo": "Blue-headed Vireo",
    "Blue Winged Warbler": "Blue-winged Warbler",
    "Boat Tailed Grackle": "Boat-tailed Grackle",
    "Brandt Cormorant": "Brandt's Cormorant",
    "Brewer Blackbird": "Brewer's Blackbird",
    "Brewer Sparrow": "Brewer's Sparrow",
    "Chestnut Sided Warbler": "Chestnut-sided Warbler",
    "Chuck Will Widow": "Chuck-will's-widow",
    "Clark Nutcracker": "Clark's Nutcracker",
    "Clay Colored Sparrow": "Clay-colored Sparrow",
    "Dark Eyed Junco": "Dark-eyed Junco",
    "Forsters Tern": "Forster's Tern",
    "Geococcyx": "Greater Roadrunner",
    "Glaucous Winged Gull": "Glaucous-winged Gull",
    "Golden Winged Warbler": "Golden-winged Warbler",
    "Gray Crowned Rosy Finch": "Gray-crowned Rosy-Finch",
    "Green Tailed Towhee": "Green-tailed Towhee",
    "Groove Billed Ani": "Groove-billed Ani",
    "Harris Sparrow": "Harris's Sparrow",
    "Heermann Gull": "Heermann's Gull",
    "Henslow Sparrow": "Henslow's Sparrow",
    "Le Conte Sparrow": "LeConte's Sparrow",
    "Long Tailed Jaeger": "Long-tailed Jaeger",
    "Nelson Sharp Tailed Sparrow": "Nelson's Sparrow",
    "Olive Sided Flycatcher": "Olive-sided Flycatcher",
    "Orange Crowned Warbler": "Orange-crowned Warbler",
    "Pied Billed Grebe": "Pied-billed Grebe",
    "Red Bellied Woodpecker": "Red-bellied Woodpecker",
    "Red Breasted Merganser": "Red-breasted Merganser",
    "Red Cockaded Woodpecker": "Red-cockaded Woodpecker",
    "Red Eyed Vireo": "Red-eyed Vireo",
    "Red Faced Cormorant": "Red-faced Cormorant",
    "Red Headed Woodpecker": "Red-headed Woodpecker",
    "Red Legged Kittiwake": "Red-legged Kittiwake",
    "Red Winged Blackbird": "Red-winged Blackbird",
    "Ring Billed Gull": "Ring-billed Gull",
    "Rose Breasted Grosbeak": "Rose-breasted Grosbeak",
    "Ruby Throated Hummingbird": "Ruby-throated Hummingbird",
    "Sayornis": "Say's Phoebe",
    "Scissor Tailed Flycatcher": "Scissor-tailed Flycatcher",
    "Scott Oriole": "Scott's Oriole",
    "Slaty Backed Gull": "Slaty-backed Gull",
    "Swainson Warbler": "Swainson's Warbler",
    "White Breasted Kingfisher": "White-breasted Kingfisher",
    "White Breasted Nuthatch": "White-breasted Nuthatch",
    "White Crowned Sparrow": "White-crowned Sparrow",
    "White Eyed Vireo": "White-eyed Vireo",
    "White Necked Raven": "White-necked Raven",
    "White Throated Sparrow": "White-throated Sparrow",
    "Wilson Warbler": "Wilson's Warbler",
    "Worm Eating Warbler": "Worm-eating Warbler",
    "Yellow Bellied Flycatcher": "Yellow-bellied Flycatcher",
    "Yellow Billed Cuckoo": "Yellow-billed Cuckoo",
    "Yellow Breasted Chat": "Yellow-breasted Chat",
    "Yellow Headed Blackbird": "Yellow-headed Blackbird",
    "Yellow Throated Vireo": "Yellow-throated Vireo"
}

def get_display_name(name):
    return DISPLAY_NAME_EXCEPTIONS.get(name, name)


# --- Load model ---
MODEL_PATH = Path("models/efficientnetb0_fine_tuned.keras")
model = tf.keras.models.load_model(MODEL_PATH)


# --- Load class mapping ---
classes_path = DATA_DIR / "classes.txt"

class_map = pd.read_csv(
    classes_path,
    sep=" ",
    header=None,
    names=["class_id", "class_name"]
)

def clean_name(name):
    # remove numeric prefix like "001."
    name = re.sub(r"^\d+\.", "", name)
    # replace underscores with spaces
    name = name.replace("_", " ")
    # tidy whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name.title()

class_id_to_name = {
    class_id - 1: clean_name(name)
    for class_id, name in zip(class_map["class_id"], class_map["class_name"])
}


# --- UI ---
st.title("🐦 BeakSpeak")
st.caption("Fine-grained bird species classifier")

# Intro
st.markdown(
    """
    Upload a bird photo and BeakSpeak will return its top predictions.

    The model was trained on **200 bird species** and works best when the bird is clearly visible.
    If the image is unclear, the bird is distant, or the species is outside the model’s known species list, predictions may be less reliable.
    """
)

# --- Search known species ---
with st.expander("Check if your bird is in the model"):
    st.caption("Search the model’s known species list.")

    search_term = st.text_input("Search species name", placeholder="e.g. Sparrow, Warbler, Albatross")

    def normalise(text):
        text = text.lower()

        # remove possessive forms: anna's -> anna
        text = re.sub(r"'s\b", "", text)

        # replace underscores / dots with spaces
        text = text.replace("_", " ").replace(".", " ")

        # remove any remaining punctuation
        text = re.sub(r"[^a-z0-9\s]", "", text)

        # collapse repeated spaces
        text = re.sub(r"\s+", " ", text).strip()

        return text

    search_term_norm = normalise(search_term)

    species_list = sorted([clean_name(name) for name in class_id_to_name.values()])

    if search_term_norm:
        matches = [
            species for species in species_list
            if search_term_norm in normalise(species)
        ]

        if matches:
            if len(matches) == 1:
                st.success(f"✔️ {get_display_name(matches[0])} is included in the model.")
            else:
                st.write(f"Found {len(matches)} matches:")
                for species in matches:
                    st.write(f"• {get_display_name(species)}")
        else:
            st.warning("No matching species found in this model.")
            st.caption("The model only recognises the species it was trained on. If the bird you're trying to identify isn’t listed, any prediction should be treated as a guess.")
    else:
        st.write(f"This model includes **{len(species_list)} species**.")

# Upload and predict
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    input_tensor = preprocess_uploaded_image(image)

    # Predict
    preds = model.predict(input_tensor)
    probs = tf.nn.softmax(preds[0]).numpy()

    # Top predictions
    top4_idx = np.argsort(probs)[-4:][::-1]

    top1_idx = top4_idx[0]
    top1_name = class_id_to_name[top1_idx]
    top1_proba = probs[top1_idx]

    top1_display_name = get_display_name(top1_name)


    with col2:
        slug = get_bird_slug(top1_name)

        # Low confidence warning
        if top1_proba < CONFIDENCE_THRESHOLD:
            st.warning("Low confidence prediction. This image may not contain a clearly identifiable bird, or it may be outside the model’s training set. Try a clearer image with the bird more centred.")

        label = (
            "high" if top1_proba > 0.75 else
            "moderate" if top1_proba > 0.55 else
            "low"
        )

        # Hero prediction
        st.markdown(f"### 🐦 Likely: **{top1_display_name}**")
        st.markdown(f"**Confidence: {top1_proba:.1%} ({label})**")
        st.markdown(f"Learn more about {top1_display_name} on [All About Birds](https://www.allaboutbirds.org/guide/{slug}/overview)")

        # Confidence interpretation
        if top1_proba > 0.75:
            st.success("High likelihood this is your bird")
        elif top1_proba > 0.55:
            st.info("Similar species are possible")
        else:
            st.caption("You may want to treat this prediction as a guess")

    # Other top predictions
    st.markdown(" ")
    st.divider()

    st.markdown("#### You might also be seeing:")

    for i in top4_idx[1:]:
        name = class_id_to_name[i]
        display_name = get_display_name(name)
        st.write(f"{display_name}: {probs[i]:.1%}")
        st.progress(float(probs[i]))

    st.caption("Confidence reflects relative likelihood across known species only. If the bird is outside the model’s training set, all predictions may be unreliable.")

    st.markdown(
    "[Explore these birds on All About Birds →](https://www.allaboutbirds.org/guide/)"
)

# Footer
st.divider()

with st.expander("🧪 How does this work?"):
    st.markdown(
        """
        **Model**

        This app uses a convolutional neural network based on EfficientNetB0, fine-tuned on the CUB-200-2011 dataset.
        The model takes an image as input and outputs a probability distribution over 200 bird species.

        **Predictions**

        The model always returns a prediction, even for unclear images or species it hasn’t seen before.
        To make this more useful, the app shows the top 3 predictions along with confidence scores.

        **Confidence threshold**

        A confidence threshold (≈0.53) is used to flag low-confidence predictions.
        This was selected using validation data to balance:
        - catching low-quality predictions
        - avoiding unnecessary warnings on correct ones

        **Limitations**

        - The model only recognises the 200 species it was trained on
        - Some species are visually very similar, which can lead to confusion
        - Performance depends heavily on image quality (clear, centred birds work best)

        **Future improvements**

        - Use part-level attributes (e.g. wing pattern, beak shape) to improve fine-grained distinctions
        - Add better handling for out-of-distribution images
        - Improve interpretability (e.g. highlighting relevant regions of the image)

        ---
        *Explore the project on [GitHub](https://github.com/keira-p/beakspeak) for more details on model training, evaluation and deployment.*
        """
    )

with st.expander("✏️ Data source and attribution"):
    st.markdown(
        """
        This application uses the [CUB-200-2011 dataset](https://authors.library.caltech.edu/records/cvm3y-5hh21).

        > Welinder, P., Branson, S., Mita, T., Wah, C., Schroff, F., Belongie, S., & Perona, P.
        > *Caltech-UCSD Birds 200*.
        > California Institute of Technology. CNS-TR-2010-001, 2010.
    """
)
