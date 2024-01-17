from django.shortcuts import render
from .forms import InputSongForm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pandas as pd
import re
import numpy as np
import contractions


def my_view(request):
    ms = pd.read_csv("../musicFinder_2.csv")

    ms = ms[ms["release_date"] >= 2000]
    ms = ms[["artist_name", "track_name", "release_date", "genre", "lyrics", "topic"]]

    stop_words = nltk.corpus.stopwords.words("english")

    def normalize_document(doc):
        # lower case and remove special characters\whitespaces
        doc = re.sub(r"[^a-zA-Z0-9\s]", "", doc, re.I | re.A)
        doc = doc.lower()
        doc = doc.strip()
        doc = contractions.fix(doc)
        # tokenize document
        tokens = nltk.word_tokenize(doc)
        # filter stopwords out of document
        filtered_tokens = [token for token in tokens if token not in stop_words]
        # re-create document from filtered tokens
        doc = " ".join(filtered_tokens)
        return doc

    normalize_corpus = np.vectorize(normalize_document)

    norm_corpus = normalize_corpus(list(ms["lyrics"]))

    tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tf.fit_transform(norm_corpus)

    doc_sim = cosine_similarity(tfidf_matrix)
    doc_sim_df = pd.DataFrame(doc_sim)

    song_list = ms["track_name"].values

    if request.method == "POST":
        form = InputSongForm(request.POST)
        if form.is_valid():
            song = form.cleaned_data["inputSong"]

            def music_finder(song_name, songs=song_list, doc_sim=doc_sim_df):
                song_id = np.where(songs == song_name)[0][0]
                song_similarities = doc_sim.iloc[song_id].values
                similar_song_id = np.argsort(-song_similarities)[1:6]
                similar_song = songs[similar_song_id]
                return similar_song

            result = music_finder(song_name=song, songs=song_list, doc_sim=doc_sim_df)
            detail = ms.loc[ms["track_name"].isin(result)]

            song_detail = []
            for index, row in detail.iterrows():
                artist_name = row["artist_name"]
                track_name = row["track_name"]
                release_date = row["release_date"]
                genre = row["genre"]
                topic = row["topic"]

                song_detail.append(
                    {
                        "artist_name": artist_name,
                        "track_name": track_name,
                        "release_date": release_date,
                        "genre": genre,
                        "topic": topic,
                    }
                )

        else:
            result = None
    else:
        form = InputSongForm()
        result = None
    return render(
        request,
        "index.html",
        {"form": form, "result": result, "song": song, "song_detail": song_detail},
    )
