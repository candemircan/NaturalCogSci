# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/feature_extractors.ipynb.

# %% auto 0
__all__ = ['extract_features', 'folder_to_word', 'get_visual_embedding', 'cleanup_temp', 'get_ada_embedding']

# %% ../nbs/feature_extractors.ipynb 2
# | code-fold: false
# | output: false
import glob
import os
from os.path import join
from shutil import rmtree
import json

import numpy as np
import pandas as pd
import torch
import fasttext
from transformers import AutoTokenizer, AutoModel
from thingsvision import get_extractor
from thingsvision.utils.data import ImageDataset, DataLoader
import tensorflow_hub as hub
import openai


from .helpers import get_project_root

# %% ../nbs/feature_extractors.ipynb 3
def extract_features(
    feature_name: str,  # same as model name. In case different encoders are available, it is in `model_encoder` format
    use_cached: bool = True,  # If `True`, rerun extraction even if the features are saved. Defaults to True.
) -> np.ndarray:  # feature array
    """
    Extract features from a model and save to disk.
    """
    project_root = get_project_root()
    temp_feature_path = join(project_root, "data", "temp", f"{feature_name}")
    final_feature_path = join(
        project_root, "data", "features", f"{feature_name.replace('/', '_')}.txt"
    )

    hugging_face_dict = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
    }

    if os.path.exists(final_feature_path) and use_cached:
        return None

    if feature_name == "task":
        objects = folder_to_word(remove_digit_underscore=False)
        ids = pd.read_csv(join(project_root, "data", "THINGS", "unique_id.csv"))[
            "id"
        ].to_list()
        weights = np.loadtxt(
            join(project_root, "data", "THINGS", "spose_embedding_49d_sorted.txt")
        )

        features = [weights[ids.index(obj), :] for obj in objects]
        features = np.array(features)

    elif feature_name == "ada-002":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        objects = folder_to_word(remove_digit_underscore=True)
        objects = [f"A photo of a {x}" for x in objects]
        features = np.array([get_ada_embedding(obj) for obj in objects])

    elif feature_name in ["bert", "roberta"]:
        objects = folder_to_word(remove_digit_underscore=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_dict[feature_name])
        model = AutoModel.from_pretrained(hugging_face_dict[feature_name]).to(device)
        objects = [f"A photo of a {x}" for x in objects]

        tokenized_objects = tokenizer(
            objects, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_objects = {
            k: torch.tensor(v).to(device) for k, v in tokenized_objects.items()
        }

        with torch.no_grad():
            latent_objects = model(**tokenized_objects)

        features = latent_objects.last_hidden_state[:, 0, :].numpy()

    elif feature_name == "fasttext":
        objects = folder_to_word(remove_digit_underscore=True)
        ft = fasttext.load_model(
            join(
                project_root,
                "data",
                "embedding_weights_and_binaries",
                "crawl-300d-2M-subword.bin",
            )
        )
        features = [ft.get_word_vector(x) for x in objects]
        features = np.array(features)
    elif feature_name == "universal_sentence_encoder":
        objects = folder_to_word(remove_digit_underscore=True)
        objects = [f"A photo of a {x}" for x in objects]
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        features = model(objects).numpy()
    else:
        features = get_visual_embedding(project_root, feature_name)

    feature_name = feature_name.replace("/", "_")
    np.savetxt(final_feature_path, features)

    return None

# %% ../nbs/feature_extractors.ipynb 4
def folder_to_word(
    remove_digit_underscore: bool,  # Remove digit and underscore from object names if true. Note that you need the digits to get the task embeddings, but not for the others. If True, the underscore gets replaced with a space.
) -> list:  # List of object names
    """
    Read file name directories and format them into words by parsing directories
    and, on demand, removing any numbers and underscores.
    """
    project_root = get_project_root()
    with open(join(project_root, "data", "features", "file_names.txt"), "r") as f:
        file_names = f.read()[:-1]  # there is an empty line in the end

    file_names = file_names.split("\n")
    file_names = [os.path.dirname(x) for x in file_names]
    file_names = [os.path.basename(x) for x in file_names]

    if remove_digit_underscore:
        file_names = ["".join([i for i in x if not i.isdigit()]) for x in file_names]
        file_names = [x.replace("_", " ") for x in file_names]
    return file_names

# %% ../nbs/feature_extractors.ipynb 5
def get_visual_embedding(
    project_root: str,  # Root directory of the project
    feature_name: str,  # Name of the feature to extract. Must be in `model_config.json`
) -> np.ndarray:  # total images by features array
    """
    Extract visual embedding using `thingsvision`
    """

    with open(join(project_root, "data", "model_configs.json")) as f:
        file = json.load(f)
    model_config = file[feature_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_parameters = None
    save_name = feature_name
    if feature_name.startswith("clip"):
        model_parameters = {"variant": feature_name.split("clip_")[1]}
        save_name = feature_name
        feature_name = "clip"
    elif feature_name.startswith("Harmonization"):
        model_parameters = {"variant": feature_name.split("Harmonization_")[1]}
        save_name = feature_name
        feature_name = "Harmonization"

    save_name = save_name.replace("/", "_")
    extractor = get_extractor(
        model_name=feature_name,
        source=model_config["source"],
        device=device,
        pretrained=True,
        model_parameters=model_parameters,
    )

    stimuli_path = join(project_root, "stimuli")
    batch_size = 1

    dataset = ImageDataset(
        root=stimuli_path,
        out_path=join(project_root, "data", "features"),
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )
    batches = DataLoader(
        dataset=dataset, batch_size=batch_size, backend=extractor.get_backend()
    )

    flatten_acts = False if feature_name.startswith("vit_") else True

    extractor.extract_features(
        batches=batches,
        module_name=model_config["module_name"],
        flatten_acts=flatten_acts,
        output_dir=join(project_root, "data", "temp", save_name),
        step_size=1,
    )

    features = cleanup_temp(project_root, save_name)

    return features

# %% ../nbs/feature_extractors.ipynb 6
def cleanup_temp(
    project_root: str,  # Root directory of the project
    save_name: str,  # name of the feature. has to match folder name under temp
) -> np.ndarray:  # total images by features array
    """
    Read features for single images from the temp folder.

    Combine them into one large array.

    If all the features are available for all images, delete the feature folder
    under temp, and return the large array.
    """

    TOTAL_IMAGES = 26107

    temp_list = glob.glob(join(project_root, "data", "temp", save_name, "*npy"))
    temp_list_sorted = sorted(
        temp_list, key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    # below we index into 0 for generality
    # it allows to extract the CLS from pytorch transformers
    # while having no effect on other embeddings, which are 1D vectors
    feature_array = np.array([np.load(x)[0, :] for x in temp_list_sorted])

    assert (
        feature_array.shape[0] == TOTAL_IMAGES
    ), f"There are features for only {feature_array.shape[0]} images.\nIt must be\
        {TOTAL_IMAGES} instead.\n temp won't be deleted and feature array won't be saved."

    return feature_array

# %% ../nbs/feature_extractors.ipynb 7
def get_ada_embedding(
    text: str,  # Sentence to be embedded
    model: str = "text-embedding-ada-002",  # Model to get embeddings from. Defaults to "text-embedding-ada-002".
) -> np.ndarray:  # word vector
    """
    Generate word embeddings from openai ada model.
    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]