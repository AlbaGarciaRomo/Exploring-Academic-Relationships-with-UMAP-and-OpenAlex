# imports

import requests
import json
import pandas as pd
import numpy as np

# umap embeddigns
import umap.umap_ as umap

# data plots
import datamapplot
import matplotlib.pyplot as plt

# sentence embedding models
from sentence_transformers import SentenceTransformer

# tf-idf encoding
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# SPECTER2
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import torch
from tqdm import tqdm

# Qualitative analysis of embeddings
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Quantitave analysis of embeddings
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Topic space
import itertools
from collections import Counter



######## ABSTRACT PROCESSING #######
# RECONSTRUCT ABSTRACT

def reconstruct_abstract(abstract_inverted_index):
    if not abstract_inverted_index:
        return "no abstract"
    # Create a dictionary mapping positions to tokens
    positions = {}
    for token, pos_list in abstract_inverted_index.items():
        for pos in pos_list:
            positions[pos] = token
    # Sort the positions and join tokens into a string
    abstract_text = " ".join(positions[i] for i in sorted(positions.keys()))
    return abstract_text

######## ABSTRACT PROCESSING #######


######## DATA FRAME BUILDING #######
# BUILD DATAFRAME DIRECTLY FROM JSON FILE (PREVIOUSLY SAVED)

def build_df(path):
    # Load the JSON data
    with open(path, "r") as file: #"works_UB_2024_all.json"
        data = json.load(file)

    # Extract works data
    works = data.get("results", [])

    # Process into a structured format
    rows = []
    for work in works:
        # Ensure primary_topic is not None before accessing its attributes
        primary_topic = work.get("primary_topic", {})
        abstract_text = reconstruct_abstract(work.get("abstract_inverted_index"))
        row = {
            "Work Name": work.get("display_name", "Unknown"),
            "Publication Year": work.get("publication_year", "Unknown"),
            "Authors": ", ".join([author["author"]["display_name"] for author in work.get("authorships", [])]),
            "Primary Topic": primary_topic.get("display_name", "Unknown") if primary_topic else "Unknown",
            "Subfield": primary_topic.get("subfield", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Field": primary_topic.get("field", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Domain": primary_topic.get("domain", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Abstract": abstract_text  # Add the reconstructed abstract
        }
        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Study the df created
    print((df.head(2)))
    print(len(df))
    # print(df["Primary Topic"].dtype)  # Should be object (string)
    # print(df["Subfield"].dtype)  
    # print(df["Field"].dtype)  
    # print(df["Domain"].dtype)
    # print(df["Abstract"].dtype)

    # print((df.isna()).sum())

    # df["Primary Topic"] = df["Primary Topic"].astype(str)
    # df["Subfield"] = df["Subfield"].astype(str)
    # df["Field"] = df["Field"].astype(str)
    # df["Domain"] = df["Domain"].astype(str)
    # df["Abstract"] = df["Abstract"].astype(str)


    # print(df["Primary Topic"].apply(type).value_counts())
    # print(df["Subfield"].apply(type).value_counts())
    # print(df["Field"].apply(type).value_counts())
    # print(df["Domain"].apply(type).value_counts())
    # print(df["Abstract"].apply(type).value_counts())

    # SAVE DATA LAYERS FOR UMAP PLOT
    # the order for the layers is from lower level to higher level according to umapplot library documentation

    np.save("openalex_layer0_cluster_labels.npy", df["Primary Topic"].values)
    np.save("openalex_layer1_cluster_labels.npy", df["Subfield"].values)
    np.save("openalex_layer2_cluster_labels.npy", df["Field"].values)
    np.save("openalex_layer3_cluster_labels.npy", df["Domain"].values)

    #Hover text (work name)
    np.save("openalex_hover_data.npy", df["Work Name"].values, allow_pickle=True)
    return df

def build_df_with_topics(path):
    # Load the JSON data
    with open(path, "r") as file:
        data = json.load(file)

    # Extract works data
    works = data.get("results", [])

    # Process into a structured format
    rows = []
    for work in works:
        primary_topic = work.get("primary_topic", {})
        topics = work.get("topics", [])
        abstract_text = reconstruct_abstract(work.get("abstract_inverted_index"))

        # Extract top 3 topics and their scores
        topic_names = [t.get("display_name", "Unknown") for t in topics]
        topic_scores = [t.get("score", 0.0) for t in topics]

        row = {
            "Work Name": work.get("display_name", "Unknown"),
            "Publication Year": work.get("publication_year", "Unknown"),
            "Authors": ", ".join([author["author"]["display_name"] for author in work.get("authorships", [])]),
            "Primary Topic": primary_topic.get("display_name", "Unknown") if primary_topic else "Unknown",
            "Subfield": primary_topic.get("subfield", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Field": primary_topic.get("field", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Domain": primary_topic.get("domain", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Abstract": abstract_text,
            "Topic 1": topic_names[0] if len(topic_names) > 0 else "None",
            "Score 1": topic_scores[0] if len(topic_scores) > 0 else 0.0,
            "Topic 2": topic_names[1] if len(topic_names) > 1 else "None",
            "Score 2": topic_scores[1] if len(topic_scores) > 1 else 0.0,
            "Topic 3": topic_names[2] if len(topic_names) > 2 else "None",
            "Score 3": topic_scores[2] if len(topic_scores) > 2 else 0.0,
        }
        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # SAVE DATA LAYERS FOR UMAP PLOT
    # the order for the layers is from lower level to higher level according to umapplot library documentation

    np.save("openalex_layer0_cluster_labels.npy", df["Primary Topic"].values)
    np.save("openalex_layer1_cluster_labels.npy", df["Subfield"].values)
    np.save("openalex_layer2_cluster_labels.npy", df["Field"].values)
    np.save("openalex_layer3_cluster_labels.npy", df["Domain"].values)

    #Hover text (work name)
    np.save("openalex_hover_data.npy", df["Work Name"].values, allow_pickle=True)

    return df



def build_df_with_topics_extended(path):
    # Load the JSON data
    with open(path, "r") as file:
        data = json.load(file)

    # Extract works data
    works = data.get("results", [])

    # Process into a structured format
    rows = []
    for work in works:
        primary_topic = work.get("primary_topic", {})
        topics = work.get("topics", [])
        abstract_text = reconstruct_abstract(work.get("abstract_inverted_index"))

        # Extract top 3 topics and their scores
        topic_names = [t.get("display_name", "Unknown") for t in topics]
        topic_scores = [t.get("score", 0.0) for t in topics]

        # Extract metadata for each topic if available
        topic_domains = [t.get("domain", {}).get("display_name", "Unknown") for t in topics]
        topic_fields = [t.get("field", {}).get("display_name", "Unknown") for t in topics]
        topic_subfields = [t.get("subfield", {}).get("display_name", "Unknown") for t in topics]

        row = {
            "Work Name": work.get("display_name", "Unknown"),
            "Publication Year": work.get("publication_year", "Unknown"),
            "Authors": ", ".join([author["author"]["display_name"] for author in work.get("authorships", [])]),
            "Primary Topic": primary_topic.get("display_name", "Unknown") if primary_topic else "Unknown",
            "Subfield": primary_topic.get("subfield", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Field": primary_topic.get("field", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Domain": primary_topic.get("domain", {}).get("display_name", "Unknown") if primary_topic else "Unknown",
            "Abstract": abstract_text,
            "Topic 1": topic_names[0] if len(topic_names) > 0 else "None",
            "Score 1": topic_scores[0] if len(topic_scores) > 0 else 0.0,
            "Domain 1": topic_domains[0] if len(topic_domains) > 0 else "Unknown",
            "Field 1": topic_fields[0] if len(topic_fields) > 0 else "Unknown",
            "Subfield 1": topic_subfields[0] if len(topic_subfields) > 0 else "Unknown",
            "Topic 2": topic_names[1] if len(topic_names) > 1 else "None",
            "Score 2": topic_scores[1] if len(topic_scores) > 1 else 0.0,
            "Domain 2": topic_domains[1] if len(topic_domains) > 1 else "Unknown",
            "Field 2": topic_fields[1] if len(topic_fields) > 1 else "Unknown",
            "Subfield 2": topic_subfields[1] if len(topic_subfields) > 1 else "Unknown",
            "Topic 3": topic_names[2] if len(topic_names) > 2 else "None",
            "Score 3": topic_scores[2] if len(topic_scores) > 2 else 0.0,
            "Domain 3": topic_domains[2] if len(topic_domains) > 2 else "Unknown",
            "Field 3": topic_fields[2] if len(topic_fields) > 2 else "Unknown",
            "Subfield 3": topic_subfields[2] if len(topic_subfields) > 2 else "Unknown",
        }
        rows.append(row)

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Save cluster labels for UMAP plotting layers
    np.save("openalex_layer0_cluster_labels.npy", df["Primary Topic"].values)
    np.save("openalex_layer1_cluster_labels.npy", df["Subfield"].values)
    np.save("openalex_layer2_cluster_labels.npy", df["Field"].values)
    np.save("openalex_layer3_cluster_labels.npy", df["Domain"].values)
    np.save("openalex_hover_data.npy", df["Work Name"].values, allow_pickle=True)

    return df


######## DATA FRAME BUILDING #######

######## EXPLORATORY ANALYSIS OF DATA #######

def plot_works_by_fields(df, uni_name_year):
    counts = df.groupby(["Domain","Field"]).size().sort_values().to_frame().reset_index()
    sns.barplot(counts, x=0, y="Field", hue="Domain", orient="h")
    # Add title
    plt.title(f"Number of Entries per Field and Domain {uni_name_year}")
    plt.show()

######## EXPLORATORY ANALYSIS OF DATA #######

######## PREPARE DATA FOR EMBEDDINGS #######

# Only passing one column of the df
def data_embeddings_from_df(df, column_name):
    # COMPUTE EMBEDDINGS
    data_embeddings =df[column_name].tolist()
    return data_embeddings

# Format title when using nomic embed model
def format_title(data_embeddings):
    prefixed_titles = [f"clustering: {title}" for title in data_embeddings]
    return prefixed_titles

# passing both title and abstract in the right format for SPECTER2 model
def format_papers(df: pd.DataFrame) -> list:
    # Ensure NaNs are replaced with empty strings
    df = df.fillna({'Work Name': '', 'Abstract': ''})

    # Rename columns for consistency
    df = df.rename(columns={'Work Name': 'title', 'Abstract': 'abstract'})

    # Convert to list of dicts
    papers = df[['title', 'abstract']].to_dict(orient='records') # list of dictionary pairs
    
    return papers

######## PREPARE DATA FOR EMBEDDINGS #######


######## COMPUTE EMBEDDINGS #######

# For TF-IDF
nltk.download('stopwords')

def custom_preprocessor(text):
    # Convert text to lowercase and remove punctuation
    text = text.lower()
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def tfidf_embeddings(language, data):
    stop_words = stopwords.words(language)
    # Adjust TF-IDF parameters to capture more nuanced features
    vectorizer = TfidfVectorizer(
        max_features=1000,          # Increase number of features
        min_df=5,                   # Term must appear in at least 5 documents
        max_df=0.9,                 # Ignore terms that appear in more than 90% of docs
        ngram_range=(1, 2),         # Capture unigrams and bigrams
        stop_words=stop_words,      # Remove common English stop words
        preprocessor=custom_preprocessor,
        sublinear_tf=True           # Apply sublinear term frequency scaling
    )

    # Assuming data_embeddings contains your work titles
    tfidf = vectorizer.fit_transform(data).toarray()
    print("TF-IDF embeddings shape:", tfidf.shape)
    return tfidf

# For sentence transformers models
def sentence_transformer_embeddings(model, data):
    model = SentenceTransformer(model, trust_remote_code=True)  
    embeddings = model.encode(data, show_progress_bar=True)
    print(embeddings)
    print(embeddings.shape)
    return embeddings

# For SPECTER2 model

# need to pass the titles and abstracts in batches to the SPECTER2 model because of memory restriction
def embed_papers_in_batches(papers, model, tokenizer, device, batch_size=32):
    """
    Embed a list of paper dicts with 'title' and 'abstract' using a SPECTER2 model in batches.
    """

    all_embeddings = []

    for i in tqdm(range(0, len(papers), batch_size)):
        batch = papers[i:i+batch_size]
        texts = [p['title'] + tokenizer.sep_token + (p.get('abstract') or '') for p in batch]

        # Tokenize and move to device
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", return_token_type_ids=False, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model(**inputs)
            embeddings = output.last_hidden_state[:, 0, :]  # CLS token
            all_embeddings.append(embeddings.cpu())  # store on CPU to avoid GPU memory overflow

    return torch.cat(all_embeddings, dim=0)

def specter2_embeddings(papers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

    #load base model
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

    #load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
    model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)
    #other possibilities: allenai/specter2_<classification|regression|adhoc_query>
    model = model.to(device)


    # Assuming `papers` is already a list of dicts with 'title' and 'abstract'
    embeddings = embed_papers_in_batches(papers, model, tokenizer, device, batch_size=32)
    print(embeddings.shape)  # Should be (n_papers, 768)
    return embeddings

######## COMPUTE EMBEDDINGS #######


######## COMPUTE DATAMAP USING UMAP #######

def create_data_map(n_neighbors, min_dist, embeddings):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42) # default n_neighbors = 15, min_dist=0.1
    data_map = reducer.fit_transform(embeddings)
    #print(data_map)
    # Save data
    np.save("openalex_data_map.npy", data_map)
    return data_map


# basic plot of 2D map
def create_simple_plot(data_map, embedding_model, n_neighbors):
    # Create the figure
    plt.figure(figsize=(10, 8))
    plt.scatter(data_map[:, 0], data_map[:, 1], label=f"Model: {embedding_model}, n_neighbors: {n_neighbors}")
    plt.title("Openalex Data Map")
    plt.legend(loc="best")

    # Save the plot to a file
    #plt.savefig(f"openalex_data_map_plot_{embedding_model}_{n_neighbors}_neighbors.png", dpi=300, bbox_inches="tight")
    plt.show()

######## COMPUTE DATAMAP USING UMAP #######

######## CREATE INTERACTIVE PLOT #######

def create_datamapplot(data_map, embedding_model, n_neighbors, university):
    #open_alex_data_map = np.load("openalex_data_map.npy", allow_pickle=True)
    open_alex_topic_layers = []
    for layer_num in range(4):
        open_alex_topic_layers.append(
        np.load(f"openalex_layer{layer_num}_cluster_labels.npy", allow_pickle=True)
        )
    openalex_hover_data = np.load("openalex_hover_data.npy", allow_pickle=True)

    # Create the interactive plot
    plot = datamapplot.create_interactive_plot(
        data_map,
        open_alex_topic_layers[1],
        open_alex_topic_layers[2],
        open_alex_topic_layers[3],
        #open_alex_topic_layers[0], # dont use the primary topic for the visualization
        hover_text = openalex_hover_data,
        initial_zoom_fraction=0.95,
        font_family="Playfair Display SC",
        title= f"OpenAlex {university} Landscape",
        sub_title= f"A data map of papers from {university} in 2024",
        #on_click="window.open(`http://google.com/search?q=\"{hover_text}\"`)",
        enable_search=True,
        darkmode=True,
        #inline_data=False,
        #offline_data_prefix="openalex_gallery",
        #cluster_boundary_polygons=True,
        #cluster_boundary_line_width=2,
    )
    plot.save(f"openalex_{university}_24_{embedding_model}_{n_neighbors}_neighbors.html")
    #print(plot)
    return plot 

def preprocess_df_extra_data(df_extended):
    """
    Process repeated fields in original df to add to hover text in final visualization
    """
    # domain colors chosen by the default of datamapplot
    domain_colors = {
        'Physical Sciences': '#2ca02c',  # green
        'Health Sciences': '#9467bd',    # purple
        'Life Sciences': '#1f77b4',      # blue
        'Social Sciences': '#d62728',    # red
        'Unknown': '#ff7f0e'             # orange
    }

    df_extra_data = df_extended[['Domain', 'Domain 2', 'Domain 3', 'Field', 'Field 2', 'Field 3']]
    df_extra_data = df_extra_data.rename(columns={'Domain': 'Domain_1', 'Domain 2': 'Domain_2', 'Domain 3': 'Domain_3', 
                                                'Field': 'Field_1', 'Field 2': 'Field_2', 'Field 3': 'Field_3'})
    
    df_extra_data["color_1"] = df_extra_data['Domain_1'].map(domain_colors)
    df_extra_data["color_2"] = df_extra_data['Domain_2'].map(domain_colors)
    df_extra_data["color_3"] = df_extra_data['Domain_3'].map(domain_colors)
    
    # 1. Field_1_New is just a copy of Field_1
    df_extra_data['Field_1_New'] = df_extra_data['Field_1']

    # 2. Field_2_New: copy Field_2 only if it differs from Field_1, else blank
    df_extra_data['Field_2_New'] = np.where(
        df_extra_data['Field_2'] != df_extra_data['Field_1'],
        df_extra_data['Field_2'],
        ''
    )

    # 3. Field_3_New: copy Field_3 only if it differs from BOTH Field_1 and Field_2, else blank
    df_extra_data['Field_3_New'] = df_extra_data.apply(
        lambda row: row['Field_3']
                    if (row['Field_3'] != row['Field_1']) and (row['Field_3'] != row['Field_2'])
                    else '',
        axis=1
    )
    return df_extra_data

def create_datamapplot_customized(data_map, df_extra_data, embedding_model, n_neighbors, university):
    #open_alex_data_map = np.load("openalex_data_map.npy", allow_pickle=True)
    open_alex_topic_layers = []
    for layer_num in range(4):
        open_alex_topic_layers.append(
        np.load(f"openalex_layer{layer_num}_cluster_labels.npy", allow_pickle=True)
        )
    #print(open_alex_topic_layers)

    openalex_hover_data = np.load("openalex_hover_data.npy", allow_pickle=True)

    badge_css = """
        border-radius:8px;
        width:fit-content;
        max-width:70%;
        margin:2px;
        padding: 2px 8px 2px 8px;
        font-size: 8pt;
    """
    hover_text_template = f"""
    <div>
        <div style="font-size:10pt;padding:2px;">{{hover_text}}</div>
        <div style="background-color:{{color_1}};color:#fff;{badge_css}">{{Field_1_New}}</div>
        <div style="background-color:{{color_2}};color:#fff;{badge_css}">{{Field_2_New}}</div>
        <div style="background-color:{{color_3}};color:#fff;{badge_css}">{{Field_3_New}}</div>
    </div>
    """

    # Create the interactive plot
    plot = datamapplot.create_interactive_plot(
        data_map,
        open_alex_topic_layers[1],
        open_alex_topic_layers[2],
        open_alex_topic_layers[3],
        # open_alex_topic_layers[0], # dont use the primary topic for the visualization
        # masked_layers[3],
        # masked_layers[2],
        # masked_layers[1],
        # hover_text = openalex_hover_data["Work Name"],
        hover_text = openalex_hover_data,
        initial_zoom_fraction=0.95,
        font_family="Playfair Display SC",
        title= f"OpenAlex {university} Landscape",
        sub_title= f"A data map of papers from {university} in 2024",
        on_click="window.open(`http://google.com/search?q=\"{hover_text}\"`)",
        enable_search=True,
        darkmode=True,
        #inline_data=False,
        #offline_data_prefix="openalex_gallery",
        # cluster_boundary_polygons=True, # boundries are not fine enough
        # cluster_boundary_line_width=1,
        # polygon_alpha=1.9
        # use_medoids=True # does not work
        # cmap = custom_cmap,
        # palette_hue_shift=-0, # rotates color palette
        # palette_hue_radius_dependence=1, # =1 is default, more hue to separate clusters
        hover_text_html_template=hover_text_template,
        extra_point_data=df_extra_data,
        histogram_data= df_extra_data['Field_1'],
        histogram_n_bins= df_extra_data['Field_1'].nunique(),
        histogram_settings={
            "histogram_title": "Primary Field",
            "histogram_width": 500,
            "histogram_height": 100,
            "histogram_log_scale": False,
            "histogram_bin_fill_color": "#6baed6",
            "histogram_bin_selected_fill_color": "#2171b5",
            "histogram_bin_unselected_fill_color": "#c6dbef"
        }
    )
    plot.save(f"openalex_{university}_24_{embedding_model}_{n_neighbors}_neighbors.html")
    #print(plot)
    return plot

######## CREATE INTERACTIVE PLOT #######


######## QUALITATIVE ANALYSIS OF EMBEDDINGS #######

def save_coordinates(df, data_map, model_name):
    # Assign UMAP coordinates back to df
    df[f'x_{model_name}'] = data_map[:, 0]
    df[f'y_{model_name}'] = data_map[:, 1]

    print(df.head(2))
    print(df.shape)

def compute_cluster_spread(df, x_col, y_col, label_col='Domain', plot=True):
    """
    For each unique label in `label_col`, compute:
    - the centroid of points in UMAP space (`x_col`, `y_col`)
    - average distance of points from their centroid (spread)
    - standard deviation of distances
    Optionally, plot clusters with centroids.
    """
    domain_colors = {
        'physical sciences': '#2ca02c',  # green
        'health sciences': '#9467bd',    # purple
        'life sciences': '#1f77b4',      # blue
        'social sciences': '#d62728',    # red
        'unknown': '#ff7f0e'             # orange
    }


    results = []
    for label in df[label_col].dropna().unique():
        group = df[df[label_col] == label]
        coords = group[[x_col, y_col]].values
        if len(coords) < 2:
            continue  # Skip small clusters

        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        avg_dist = dists.mean()
        std_dev = dists.std()

        results.append({
            'label': label,
            'count': len(coords),
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'avg_dist': avg_dist,
            'std_dev': std_dev
        })

        if plot:
            # Get color from map, fallback to gray if label not recognized
            c = domain_colors.get(label.lower(), 'gray')
            plt.scatter(coords[:, 0], coords[:, 1], s=10, label=label, alpha=0.4, color=c)
            plt.scatter(*centroid, c='black', s=50, marker='x')
            plt.text(centroid[0], centroid[1], label, fontsize=8, ha='center', va='center', color='black')

    if plot:
        plt.title(f"UMAP clusters by {label_col}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).sort_values(by='avg_dist')

def compute_cluster_spread_with_density(df, x_col, y_col, label_col='Domain', plot=True, add_kde=True):
    """
    For each unique label in `label_col`, compute:
    - the centroid of points in UMAP space (`x_col`, `y_col`)
    - average distance of points from their centroid (spread)
    - standard deviation of distances
    Optionally, plot clusters with centroids and KDE overlay.
    """

    results = []

    if plot:
        plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Optional KDE overlay (over all points, not per cluster)
        if add_kde:
            sns.kdeplot(
                data=df,
                x=x_col,
                y=y_col,
                fill=True,
                cmap="Reds",
                alpha=0.3,
                thresh=0.05,
                levels=100,
                ax=ax
            )

    for label in df[label_col].dropna().unique():
        group = df[df[label_col] == label]
        coords = group[[x_col, y_col]].values
        if len(coords) < 2:
            continue  # Skip small clusters

        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        avg_dist = dists.mean()
        std_dev = dists.std()

        results.append({
            'label': label,
            'count': len(coords),
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'avg_dist': avg_dist,
            'std_dev': std_dev
        })

        if plot:
            plt.scatter(coords[:, 0], coords[:, 1], s=10, label=label, alpha=0.4)
            plt.scatter(*centroid, c='black', s=50, marker='x')
            plt.text(centroid[0], centroid[1], label, fontsize=8, ha='center', va='center', color='black')

    if plot:
        plt.title(f"UMAP Clusters by {label_col} with KDE Overlay")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).sort_values(by='avg_dist')



def compute_cluster_spread_embeddings(df, embeddings, label_col='Domain'):
    """
    Computes average distance and std dev to centroid in the original embedding space.
    
    Parameters:
    - df: DataFrame with metadata (must contain label_col and id_col)
    - embeddings: np.array of shape (n_samples, n_dims)
    - label_col: Column name in df for the category (e.g. 'domain')
    - id_col: Column in df used to align with embeddings (must match the order of embeddings)
    """
    # Make sure the index matches the embeddings
    if len(df) != len(embeddings):
        raise ValueError("Mismatch between number of rows in df and embeddings.")

    results = []
    for label in df[label_col].dropna().unique():
        group_idx = df[df[label_col] == label].index
        if len(group_idx) < 2:
            continue  # Skip very small groups

        group_embeddings = embeddings[group_idx]
        centroid = group_embeddings.mean(axis=0)
        dists = np.linalg.norm(group_embeddings - centroid, axis=1)
        avg_dist = dists.mean()
        std_dev = dists.std()

        results.append({
            'label': label,
            'count': len(group_idx),
            'avg_dist': avg_dist,
            'std_dev': std_dev,
            'centroid': centroid
        })

    return pd.DataFrame(results).sort_values(by='avg_dist')

def compute_cluster_spread_with_authors(df, x_col, y_col, x_col_auth, y_col_auth, label_col='Domain', plot=True, author_centroids=None):
    """
    For each unique label in `label_col`, compute:
    - the centroid of points in UMAP space (`x_col`, `y_col`)
    - average distance of points from their centroid (spread)
    - standard deviation of distances
    Optionally, plot clusters with centroids and author centroids.
    """
    domain_colors = {
        'physical sciences': '#2ca02c',  # green
        'health sciences': '#9467bd',    # purple
        'life sciences': '#1f77b4',      # blue
        'social sciences': '#d62728',    # red
        'unknown': '#ff7f0e'             # orange
    }
    
    results = []
    for label in df[label_col].dropna().unique():
        group = df[df[label_col] == label]
        coords = group[[x_col, y_col]].values
        if len(coords) < 2:
            continue  # Skip small clusters

        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        avg_dist = dists.mean()
        std_dev = dists.std()

        results.append({
            'label': label,
            'count': len(coords),
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'avg_dist': avg_dist,
            'std_dev': std_dev
        })

        if plot:
            c = domain_colors.get(label.lower(), 'gray')
            plt.scatter(coords[:, 0], coords[:, 1], s=10, label=label, alpha=0.4, color=c)
            plt.scatter(*centroid, c='black', s=50, marker='x')
            plt.text(centroid[0], centroid[1], label, fontsize=8, ha='center', va='center', color='black')

    if plot:
        # Plot author centroids if provided
        if author_centroids is not None:
            plt.scatter(
                author_centroids[x_col_auth],
                author_centroids[y_col_auth],
                s=5,
                c='black',
                marker='*',
                label='Author Centroids'
            )

        plt.title(f"UMAP clusters by {label_col}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).sort_values(by='avg_dist')

######## QUALITATIVE ANALYSIS OF EMBEDDINGS #######

######## QUANTITATIVE ANALYSIS OF EMBEDDINGS #######

def clustering_metrics(df, models, label_column='Domain'):
    """
    Evaluate clustering metrics (unsupervised) for a list of 2D embedding models.

    Parameters:
    - df: pandas DataFrame containing columns x_<model> and y_<model>
    - models: list of strings with model suffixes (e.g., ['st', 'nomic', 'specter'])
    - label_column: column name used as ground truth labels for evaluation

    Returns:
    - results: a dictionary of metrics for each model
    """
    labels = df[label_column].astype(str).values
    results = {}

    for model in models:
        print(f"\nResults for model: {model}")
        X = df[[f"x_{model}", f"y_{model}"]].values

        # Compute metrics
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)

        # Print and store results
        print(f"  Silhouette Score:       {silhouette:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index:   {davies_bouldin:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index:{calinski_harabasz:.4f} (higher is better)")

        results[model] = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz
        }

    return results

######## QUANTITATIVE ANALYSIS OF EMBEDDINGS #######

######## EMBEDDINGS FOR TOPIC SPACE #######

# Compute domain, field, subfield and topic embeddings

def build_topic_presence_matrix(df):

    # Step 1: Define all label-related columns
    topic_cols = ['Topic 1', 'Topic 2', 'Topic 3']
    subfield_cols = ['Subfield 1', 'Subfield 2', 'Subfield 3']
    field_cols = ['Field 1', 'Field 2', 'Field 3']
    domain_cols = ['Domain 1', 'Domain 2', 'Domain 3']

    # Step 2: Collect all unique values per category
    def clean(values):
        return [v for v in pd.unique(values) if v not in (None, "None", "Unknown")]

    all_topics = clean(df[topic_cols].values.ravel())
    all_subfields = clean(df[subfield_cols].values.ravel())
    all_fields = clean(df[field_cols].values.ravel())
    all_domains = clean(df[domain_cols].values.ravel())

    # Step 3: Create the full list of features
    feature_names = (
        ['TOPIC_' + t for t in all_topics] +
        ['SUBFIELD_' + s for s in all_subfields] +
        ['FIELD_' + f for f in all_fields] +
        ['DOMAIN_' + d for d in all_domains]
    )
    feature_to_index = {name: idx for idx, name in enumerate(feature_names)}

    # Step 4: Initialize count matrix
    num_works = len(df)
    num_features = len(feature_names)
    matrix = np.zeros((num_works, num_features))

    # Step 5: Fill in counts instead of binary flags
    for i, row in df.iterrows():
        for tcol in topic_cols:
            topic = row[tcol]
            if topic not in (None, "None", "Unknown"):
                fname = 'TOPIC_' + topic
                matrix[i, feature_to_index[fname]] += 1

        for scol in subfield_cols:
            subfield = row[scol]
            if subfield not in (None, "None", "Unknown"):
                fname = 'SUBFIELD_' + subfield
                matrix[i, feature_to_index[fname]] += 1

        for fcol in field_cols:
            field = row[fcol]
            if field not in (None, "None", "Unknown"):
                fname = 'FIELD_' + field
                matrix[i, feature_to_index[fname]] += 1

        for dcol in domain_cols:
            domain = row[dcol]
            if domain not in (None, "None", "Unknown"):
                fname = 'DOMAIN_' + domain
                matrix[i, feature_to_index[fname]] += 1

    # Step 6: Wrap in DataFrame
    matrix_df = pd.DataFrame(matrix, columns=feature_names)
    return matrix_df

# Create cluster visualization with all combinations of different domains

def compute_cluster_spread_multidomain(df, x_col, y_col, domain_cols=['Domain 1', 'Domain 2', 'Domain 3'], plot=True):

    # Create composite domain label per work (sorted and joined)
    def combine_domains(row):
        domains = [row[col] for col in domain_cols if row[col] not in (None, "None", "Unknown")]
        #domains = [row[col] for col in domain_cols]
        return " + ".join(sorted(set(domains))) if domains else None

    df = df.copy()
    df['Combined_Domains'] = df.apply(combine_domains, axis=1)

    results = []
    for label in df['Combined_Domains'].dropna().unique():
        group = df[df['Combined_Domains'] == label]
        coords = group[[x_col, y_col]].values
        if len(coords) < 2:
            continue

        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        avg_dist = dists.mean()
        std_dev = dists.std()

        results.append({
            'label': label,
            'count': len(coords),
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'avg_dist': avg_dist,
            'std_dev': std_dev
        })

        if plot:
            plt.scatter(coords[:, 0], coords[:, 1], s=10, label=label, alpha=0.4)
            plt.scatter(*centroid, c='black', s=50, marker='x')
            #plt.text(centroid[0], centroid[1], label, fontsize=6, ha='center', va='center', color='black')

    if plot:
        plt.title("UMAP clusters by Combined Domains")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='xx-small')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).sort_values(by='avg_dist')

# Check different domain combinations between topics

## COUNT HOW MANY VECTORS HAVE MORE THAN ONE DIFFERENT DOMAIN

def detect_cross_domain_vectors(matrix_df):
    # Step 1: Identify domain-related columns
    domain_cols = [col for col in matrix_df.columns if col.startswith("DOMAIN_")]

    # Step 2: Count how many domain columns are non-zero per row
    domain_counts = (matrix_df[domain_cols] > 0).sum(axis=1)

    # Step 3: Return boolean mask or indices of rows with more than one domain
    cross_domain_mask = domain_counts > 1
    cross_domain_indices = matrix_df.index[cross_domain_mask].tolist()

    return cross_domain_indices, matrix_df.loc[cross_domain_mask]

## COUNT HOW MANY VECTORS HAVE THE SAME DOMAIN FOR ALL THREE TOPICS

def count_same_domain_triplets(matrix_df):
    # Step 1: Get all domain-related columns
    domain_cols = [col for col in matrix_df.columns if col.startswith("DOMAIN_")]

    # Step 2: Identify cross-domain rows using your function
    domain_counts = (matrix_df[domain_cols] > 0).sum(axis=1)
    cross_domain_mask = domain_counts > 1

    # Step 3: Check remaining (non-cross-domain) rows
    single_domain_df = matrix_df[~cross_domain_mask]

    return single_domain_df, len(single_domain_df)

## Cooccurrence matrix

def compute_domain_cooccurrence(df):
    domain_cols = ['Domain 1', 'Domain 2', 'Domain 3']
    
    # Get all unique valid domains
    all_domains = pd.unique(df[domain_cols].values.ravel())
    all_domains = [d for d in all_domains if d not in (None, "None", "Unknown")]

    # Initialize co-occurrence matrix
    co_matrix = pd.DataFrame(0, index=all_domains, columns=all_domains, dtype=int)

    for _, row in df.iterrows():
        row_domains = [row[col] for col in domain_cols if row[col] not in (None, "None", "Unknown")]
        for d1, d2 in itertools.combinations(set(row_domains), 2):
            co_matrix.at[d1, d2] += 1
            co_matrix.at[d2, d1] += 1  # Symmetric

    return co_matrix


## Count all domain combinations
def count_all_domain_combinations(df, domain_cols=['Domain 1', 'Domain 2', 'Domain 3']):
    combo_counter = Counter()

    for _, row in df.iterrows():
        #domains = [row[col] for col in domain_cols if row[col] not in (None, "None", "Unknown")]
        domains = [row[col] for col in domain_cols]
        
        if domains:
            combo = tuple(sorted(set(domains)))  # Sorted to avoid treating same combos differently
            combo_counter[combo] += 1

    # Convert to a sorted DataFrame for display
    combo_df = pd.DataFrame([
        {"Domain Combination": " + ".join(combo), "Count": count}
        for combo, count in combo_counter.items()
    ]).sort_values(by="Count", ascending=False).reset_index(drop=True)

    return combo_df

## Plot domain combinations
def plot_domain_combinations(combo_df, top_n=20):
    top_combos = combo_df.head(top_n)
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_combos["Domain Combination"], top_combos["Count"], color='skyblue')
    plt.xlabel("Number of Works")
    plt.title(f"Domain Combinations")
    plt.gca().invert_yaxis()  # Highest on top

    # Add labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 5, bar.get_y() + bar.get_height()/2,
                 f'{int(width)}', va='center')

    plt.tight_layout()
    plt.show()


######## EMBEDDINGS FOR TOPIC SPACE #######

######## AUTHOR ANALYSIS #######

def create_exploded_authors_df(df):
    # format authors into a list
    df['Authors'] = df['Authors'].fillna("")

    # 2. Apply a function that:
    #    Splits the string on commas
    #    Strips leading/trailing whitespace from each piece
    #    Drops any empty remnants (in case of “, ,” or trailing commas)
    df['Author_list'] = df['Authors'].apply(lambda s: [a.strip() for a in s.split(',') if a.strip()])
    #print(df)

    # count num of author for each paper
    df['num_authors'] = df['Author_list'].apply(len)
    #print(df)

    # filter out works with more than 20 authors
    df_filtered = df[df['num_authors'] <= 20].copy()
    #print(df_filtered)


    # .explode() so that each (paper, single author) becomes its own row
    df_exploded = df_filtered.explode('Author_list').rename(columns={'Author_list': 'author'})
    print(df_exploded.head(10))
    return df_exploded


def plot_num_authors(df, university):
    # df must have a column called 'num_authors'
    counts_authors = (
    df
    .groupby("num_authors")
    .size()
    .to_frame(name="count")
    .reset_index())

    # 2. Sort by num_authors descending:
    counts_authors = counts_authors.sort_values("num_authors", ascending=False)

    # 3. Create a vertical bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=counts_authors,
        x="num_authors",
        y="count",
        #palette="Blues_d",
        ax=ax
    )

    # 4. Labels & title
    ax.set_xlabel("Number of Authors")
    ax.set_ylabel("Number of Papers")
    ax.set_title(f"Distribution of # of Authors per Paper ({university} 2024)")

    # 5. Set x‐ticks every 5 from 1 to 100
    ticks = np.arange(0, 100, 5)
    ax.set_xticks(ticks)
    # ax.set_xticklabels(ticks, rotation=0)

    # 6. Force the x‐axis limits to cover 1 through 100
    ax.set_xlim(-1, 100)

    plt.tight_layout()
    plt.show()

# Group by author and compute the mean embedding

# Helper to compute the mean embedding for a column
def mean_embedding(group, column):
    return np.mean(np.stack(group[column].values), axis=0)

def compute_mean_embeddings_by_author(df_exploded, embedding_cols):
    """
    Compute the mean embeddings per author from exploded paper-author dataframe.

    Parameters:
        df_exploded (pd.DataFrame): A DataFrame where each row corresponds to a paper-author pair.
        embedding_cols (list of str): List of column names containing embeddings.

    Returns:
        pd.DataFrame: A DataFrame with one row per author and mean embeddings.
    """

    # Ensure all embedding columns are NumPy arrays
    for col in embedding_cols:
        df_exploded[col] = df_exploded[col].apply(lambda x: np.array(x))

    # Build the authors dataframe
    authors_df = df_exploded.groupby('author').apply(
        lambda group: pd.Series({
            f'mean_{col}': mean_embedding(group, col) for col in embedding_cols
        } | {
            'num_papers': len(group)
        })
    ).reset_index()

    return authors_df

# modification of function cluster spread with authors

def compute_cluster_spread_with_authors(df, x_col, y_col, x_col_auth, y_col_auth, label_col='Domain', plot=True, author_centroids=None):
    """
    For each unique label in `label_col`, compute:
    - the centroid of points in UMAP space (`x_col`, `y_col`)
    - average distance of points from their centroid (spread)
    - standard deviation of distances
    Optionally, plot clusters with centroids and author centroids.
    """
    results = []
    for label in df[label_col].dropna().unique():
        group = df[df[label_col] == label]
        coords = group[[x_col, y_col]].values
        if len(coords) < 2:
            continue  # Skip small clusters

        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        avg_dist = dists.mean()
        std_dev = dists.std()

        results.append({
            'label': label,
            'count': len(coords),
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'avg_dist': avg_dist,
            'std_dev': std_dev
        })

        if plot:
            plt.scatter(coords[:, 0], coords[:, 1], s=10, label=label, alpha=0.4)
            plt.scatter(*centroid, c='black', s=50, marker='x')
            plt.text(centroid[0], centroid[1], label, fontsize=8, ha='center', va='center', color='black')

    if plot:
        # Plot author centroids if provided
        if author_centroids is not None:
            plt.scatter(
                author_centroids[x_col_auth],
                author_centroids[y_col_auth],
                s=5,
                c='black',
                marker='*',
                label='Author Centroids'
            )

        plt.title(f"UMAP clusters by {label_col}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results).sort_values(by='avg_dist')


def create_author_datamapplot_2(df, model_name, n_neighbors):
    "Visualization authors"
    coords = df[[f"x_authors_{model_name}", f"y_authors_{model_name}"]].values

    # Label for histogram and filter/search
    domain_labels = np.array([", ".join(d) if isinstance(d, list) else str(d) for d in df['Domain'].tolist()])

    # Label used for coloring
    university_labels = np.array(df['institution'].astype(str).tolist())

    # === Define color mapping ===
    color_mapping = {
        "UB": "#2a8bb4",       # Blue
        "Utrecht": "#d62728",  # Red
        "Both": "#ffdd57"      # Yellow
    }

    # Create marker_color_array
    marker_color_array = pd.Series(university_labels).map(color_mapping).fillna("#cccccc").values

    # Hover text
    hover_text = np.array(df['author'].astype(str).tolist())
    hover_text_template = """
    <div style="font-size:10pt; padding:2px;">
        <strong>{author}</strong><br>
        Number of papers: {num_papers}
    </div>
    """

    # === Create custom legend HTML & CSS ===
    custom_css = custom_css = """
    #legend {
        position: absolute;
        top: 200px;
        left: 28px;
        z-index: 2;
        padding: 10px 14px;
        border: 1px solid #ccc;
        border-radius: 6px;
        font-family: sans-serif;
        font-size: 14px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.15);
    }

    #legend-title {
        font-weight: bold;
        font-size: 15px;
        margin-bottom: 10px;  /* <-- This adds space below the title */
    }

    .row {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
    }

    .box {
        height: 14px;
        width: 14px;
        border-radius: 2px;
        margin-right: 8px;
        background-color: #000;
    }
    """

    custom_html = """
    <div id="legend" class="container-box">
        <div id="legend-title">Institution</div>
    """
    for field, color in color_mapping.items():
        custom_html += f'    <div class="row"><div class="box" style="background-color:{color};"></div>{field}</div>\n'
    custom_html += "</div>"


    # Create the plot
    plot = datamapplot.create_interactive_plot(
        coords,
        domain_labels,                         # Used for filtering/search
        hover_text=hover_text,
        hover_text_html_template=hover_text_template,
        extra_point_data=df,
        initial_zoom_fraction=0.99,
        font_family="Playfair Display SC",
        title="Author Landscape",
        sub_title="UB vs Utrecht University",
        on_click="window.open(`http://google.com/search?q=\"{hover_text}\"`)",
        enable_search=True,
        darkmode=True,
        histogram_data=domain_labels,
        histogram_n_bins=len(set(domain_labels)),
        marker_color_array=marker_color_array,                # Coloring by institution
        histogram_settings={
            "histogram_title": "Primary Domain",
            "histogram_width": 500,
            "histogram_height": 100,
            "histogram_log_scale": False,
            "histogram_bin_fill_color": "#6baed6",
            "histogram_bin_selected_fill_color": "#2171b5",
            "histogram_bin_unselected_fill_color": "#c6dbef"
        },
        custom_css=custom_css,
        custom_html=custom_html,
        color_label_text=False
    )

    plot.save(f"TEST_UB_vs_Utrecht_authors_{model_name}_{n_neighbors}_byInstitution.html")
    return plot


######## AUTHOR ANALYSIS #######




