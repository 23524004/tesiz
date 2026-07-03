import pandas as pd
import numpy as np
import yake
from collections import defaultdict
import networkx as nx
import itertools
import matplotlib.pyplot as plt
import re

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary


class TopicLabeling:
    """
    YAKE (phrase extraction)
            ↓
    Phrase-level Graph
            ↓
    TextRank (PageRank)
    """

    def __init__(
        self, 
        ### YAKE PARAMS
        n_keywords=15, 
        n_gram=3, 
        deduplim=0.9, 
        domain_stopwords=None,
        yake_multiplier=3,
        filter_generic_phrases=False,
        ngram_mode="mixed",             # "mixed" atau "exact"
        
        top_n=5, 
        
        debug=False
    ):
        """
        Parameters
        ----------
        -- YAKE --
        n_keywords : int
            Jumlah kandidat phrase final yang ingin dipertahankan
            setelah proses filtering domain-generic.
    
        n_gram : int
            Maksimum jumlah kata dalam satu keyphrase.
            (e.g., 2 yakni unigrams & bigrams)
    
        deduplim : float
            Threshold untuk menghilangkan keyword yang mirip.
            Semakin besar → semakin sedikit frasa yang dianggap duplikat , lebih distinct
    
        yake_multiplier : int
            Faktor pengali untuk strategi over-generation.
            YAKE akan menghasilkan:
                n_keywords * yake_multiplier kandidat awal.
            Setelah filtering domain-generic dilakukan,
            sistem akan mengambil n_keywords teratas.
            Tujuannya agar jumlah kandidat final tetap stabil
            meskipun beberapa frasa terfilter.
    
        domain_stopwords : set or list, optional
            Daftar kata generik domain (misal: "study", "method",
            "result", dll) yang sering muncul pada dokumen ilmiah
            namun kurang informatif sebagai label topik.
    
            Phrase yang:
                - seluruh katanya generik, atau
                - lebih dari 50% katanya generik
                  akan dibuang sebelum masuk ke tahap graph.
    
            Jika None, sistem menggunakan daftar default
            domain stopword akademik bahasa Inggris.
    
        -- TextRank --
        top_n : int
            Jumlah label akhir yang dipilih berdasarkan
            skor PageRank tertinggi.
    
        -- LainLain --
        debug : bool
            Jika True, menyimpan informasi debugging
            (similarity matrix, graph edges, pagerank score, dll).
        """

        self.n_keywords = n_keywords
        self.deduplim = deduplim
        self.n_gram = n_gram
        self.yake_multiplier = yake_multiplier
        self.yake_multiplier = yake_multiplier
        default_domain_stopwords = {
            "study", "paper", "research", "result", "results",
            "method", "methods", "approach", "analysis",
            "model", "models", "proposed", "based",
            "using", "new", "novel", "effect",
            "effects", "performance", "system",
            "data", "dataset", "evaluation"
        }
        self.domain_stopwords = set(domain_stopwords) if domain_stopwords else default_domain_stopwords
        self.filter_generic_phrases = filter_generic_phrases
        self.ngram_mode = ngram_mode

        self.top_n = top_n
        
        self.debug = debug

        self.cluster_labels = {}
        self.debug_info = {}

    # ==========================================================
    # PUBLIC METHODS
    # ==========================================================

    def fit(self, df):

        if not {'title', 'abstract', 'cluster'}.issubset(df.columns):
            raise ValueError("DataFrame harus memiliki kolom: 'title', 'abstract', 'cluster'")

        self.cluster_labels = {}
        self.debug_info = {}

        df = df.copy()
        df['dokumen'] = self._prep_data(df)

        grouped = df.groupby('cluster')

        for cluster_id, group in grouped:
            docs = group['dokumen'].tolist()

            # 1️⃣ YAKE
            kandidat_frasa, gabung_text, yake_score = self._extract_yake(docs)
            if not kandidat_frasa:
                self.cluster_labels[cluster_id] = "No label found"
                continue

            # Debug isi variabel per cluster
            if self.debug:
                self.debug_info[cluster_id] = {
                    "input_to_yake": None,
                    "yake_output": None,
                    "yake_output_score": None,
                    "present_phrases_per_doc": [],
                    "graph_edges": None,
                    "pagerank_scores": None
                }
            if self.debug:
                self.debug_info[cluster_id]["input_to_yake"] = gabung_text
                self.debug_info[cluster_id]["yake_output"] = kandidat_frasa
                self.debug_info[cluster_id]["yake_output_score"] = yake_score



            # 2️⃣ Build graph
            phrase_graph = self._build_phrase_graph(
                docs, kandidat_frasa, cluster_id
            )
            if phrase_graph.number_of_nodes() == 0:
                self.cluster_labels[cluster_id] = "No label found"
                continue




            # 3️⃣ TextRank
            final_phrases, ranks = self._textrank(phrase_graph)
            self.cluster_labels[cluster_id] = ", ".join(final_phrases)

            if self.debug:
                self.debug_info[cluster_id]["pagerank_scores"] = ranks
                self.debug_info[cluster_id]["graph_edges"] = list(
                    phrase_graph.edges(data=True)
                )

            



        return self


    
    def predict(self):
        return self.cluster_labels


    def evaluate_topic_coherence(self, df, method="c_v"):
        """
        Hitung topic coherence untuk setiap cluster label.
    
        Parameters
        ----------
        df : DataFrame
            DataFrame input yang sama seperti pada fit()
        method : str
            "c_v" atau "c_npmi"
            default = "c_v"
    
        Returns
        -------
        DataFrame
            Kolom:
            cluster
            label
            coherence
        """
        if method not in ["c_v", "c_npmi"]:
            raise ValueError("method harus 'c_v' atau 'c_npmi'")
    
        results = []
    
        df = df.copy()
        df["dokumen"] = self._prep_data(df)
    
        grouped = df.groupby("cluster")
    
        for cluster_id, group in grouped:
    
            docs = group["dokumen"].tolist()
    
            label_str = self.cluster_labels.get(cluster_id)
    
            if not label_str:
                results.append({
                    "cluster": cluster_id,
                    "label": None,
                    "coherence": None
                })
                continue
    
            texts = [doc.lower().split() for doc in docs]
    
            dictionary = Dictionary(texts)
    
            labels = [l.strip() for l in label_str.split(",")]
    
            # flatten words dari label
            topic_words = list(set(
                w for label in labels for w in label.lower().split()
            ))
    
            if len(topic_words) < 2:
                coherence = None
            else:
    
                topics = [topic_words]
    
                cm = CoherenceModel(
                    topics=topics,
                    texts=texts,
                    dictionary=dictionary,
                    coherence=method
                )
    
                coherence = cm.get_coherence()
    
            results.append({
                "cluster": cluster_id,
                "label": label_str,
                "coherence": coherence,
                "n_docs": len(docs),
                "n_words": len(topic_words)
            })
    
        return pd.DataFrame(results)
        # scores = labeler.evaluate_topic_coherence(df)                  ; print(scores)
        # scores = labeler.evaluate_topic_coherence(df,method="c_npmi")  ; print(scores)


    # def evaluate_cluster_labels(self, df, method="c_v"):
    #     from gensim.models.coherencemodel import CoherenceModel
    #     from gensim.corpora import Dictionary
    #     results = []
    #     df = df.copy()
    #     df["dokumen"] = self._prep_data(df)
    #     grouped = df.groupby("cluster")
    #     all_docs = df["dokumen"].str.lower().tolist()
    #     for cluster_id, group in grouped:
    #         docs = group["dokumen"].str.lower().tolist()
    #         label_str = self.cluster_labels.get(cluster_id)
    #         if not label_str:
    #             continue
    #         labels = [l.strip() for l in label_str.split(",")]
    #         # ----------------
    #         # 1. Coherence
    #         # ----------------
    #         texts = [doc.split() for doc in docs]
    #         dictionary = Dictionary(texts)
    #         topics = [label.split() for label in labels]
    #         cm = CoherenceModel(
    #             topics=topics,
    #             texts=texts,
    #             dictionary=dictionary,
    #             coherence=method
    #         )
    #         coherence = cm.get_coherence()
    #         # ----------------
    #         # 2. Coverage
    #         # ----------------
    #         doc_hits = 0
    #         for doc in docs:
    #             if any(label in doc for label in labels):
    #                 doc_hits += 1
    #         coverage = doc_hits / len(docs)
    #         # ----------------
    #         # 3. Distinctiveness
    #         # ----------------
    #         cluster_freq = sum(
    #             doc.count(label)
    #             for doc in docs
    #             for label in labels
    #         )
    #         global_freq = sum(
    #             doc.count(label)
    #             for doc in all_docs
    #             for label in labels
    #         )
    #         if global_freq == 0:
    #             distinctiveness = 0
    #         else:
    #             distinctiveness = cluster_freq / global_freq
    #         # ----------------
    #         # 4. Label Quality Score
    #         # ----------------
    #         lqs = (
    #             0.4 * coherence +
    #             0.3 * coverage +
    #             0.3 * distinctiveness
    #         )
    #         results.append({
    #             "cluster": cluster_id,
    #             "size": len(docs),
    #             "label": label_str,
    #             "coherence": coherence,
    #             "coverage": coverage,
    #             "distinctiveness": distinctiveness,
    #             "label_quality_score": lqs
    #         })
    #     return pd.DataFrame(results)

    

    def get_debug_info(self):
        return self.debug_info

    def print_debug(self, cluster_id):
        """
        Print detail debugging per cluster
        """
        if not self.debug:
            print("Debug mode is OFF")
            return

        if cluster_id not in self.debug_info:
            print("Cluster tidak ditemukan")
            return

        info = self.debug_info[cluster_id]

        print("\n===== DEBUG CLUSTER", cluster_id, "=====")
        print("\n--- Input ke YAKE ---")
        print(info["input_to_yake"])

        print("\n--- Output YAKE ---")
        print(info["yake_output"])

        print("\n--- Phrase per Dokumen ---")
        for i, phrases in enumerate(info["present_phrases_per_doc"]):
            print(f"Doc {i}:", phrases)

        print("\n--- Graph Edges (p1, p2, weight) ---")
        print(info["graph_edges"])

        print("\n--- PageRank Scores ---")
        print(info["pagerank_scores"])

    def plot_graph(self, cluster_id):
        """
        Visualisasi graph cluster tertentu
        """
        if cluster_id not in self.debug_info:
            print("Cluster tidak ditemukan")
            return

        edges = self.debug_info[cluster_id]["graph_edges"]
        if not edges:
            print("Graph kosong")
            return

        G = nx.Graph()
        G.add_edges_from(edges)

        # pos = nx.spring_layout(G, seed=42)
        # weights = [G[u][v]['weight'] for u, v in G.edges()]

        # plt.figure(figsize=(8, 6))
        # nx.draw(G, pos, with_labels=True, node_size=2000)
        # nx.draw_networkx_edge_labels(
        #     G, pos,
        #     edge_labels={(u, v): G[u][v]['weight'] for u, v in G.edges()}
        # )
        # plt.title(f"Phrase Graph Cluster {cluster_id}")
        # plt.show()

        # SOLUSI: Naikkan nilai k biar ruang lebih luas
        pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42)
    
        plt.figure(figsize=(12, 8)) # Perbesar kanvas !!!
    
        nx.draw(
            G, pos, 
            with_labels=True, 
            node_size=1500,        # mainkan node size !!!
            node_color='#347ab0', 
            font_size=10,          # mainkan font !!!
            font_weight='bold',
            edge_color='gray'
        )

        edge_labels = {(u, v): G[u][v]['weight'] for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
        plt.title(f"Phrase Graph Cluster {cluster_id}")
        plt.show()


    
    # ==========================================================
    # BASELINE
    # ==========================================================
    def baseline_yake_only_coherence(self, df, method="c_v"):
        """
        Evaluasi topic coherence menggunakan YAKE saja (tanpa TextRank).
    
        Pipeline:
        docs → YAKE → top keywords → coherence
    
        Parameters
        ----------
        df : DataFrame
            Harus memiliki kolom: title, abstract, cluster
        method : str
            "c_v" atau "c_npmi"
    
        Returns
        -------
        DataFrame:
            cluster, label, coherence, n_docs, n_words


        Usage
        -------
        labeler = TopicLabeling(
        n_keywords=10,
        n_gram=3,
        deduplim=0.9,
        filter_generic_phrases=True
        )
        
        df_scores = labeler.baseline_yake_only_coherence(df)
        
        print(df_scores)
        """
    
        if method not in ["c_v", "c_npmi"]:
            raise ValueError("method harus 'c_v' atau 'c_npmi'")
    
        results = []
    
        df = df.copy()
        df["dokumen"] = self._prep_data(df)
    
        grouped = df.groupby("cluster")
    
        for cluster_id, group in grouped:
    
            docs = group["dokumen"].tolist()
    
            # 🔹 YAKE ONLY
            phrases, _, _ = self._extract_yake(docs)
    
            if not phrases:
                results.append({
                    "cluster": cluster_id,
                    "label": None,
                    "coherence": None,
                    "n_docs": len(docs),
                    "n_words": 0
                })
                continue
    
            label_str = ", ".join(phrases)
    
            # 🔹 PREPARE COHERENCE INPUT
            texts = [doc.lower().split() for doc in docs]
            dictionary = Dictionary(texts)
    
            topic_words = list(set(
                w for phrase in phrases for w in phrase.split()
            ))
    
            if len(topic_words) < 2:
                coherence = None
            else:
                cm = CoherenceModel(
                    topics=[topic_words],
                    texts=texts,
                    dictionary=dictionary,
                    coherence=method
                )
                coherence = cm.get_coherence()
    
            results.append({
                "cluster": cluster_id,
                "label": label_str,
                "coherence": coherence,
                "n_docs": len(docs),
                "n_words": len(topic_words)
            })
    
        return pd.DataFrame(results)

    

    # ==========================================================
    # INTERNAL METHODS
    # ==========================================================

    def _prep_data(self, df):
        return ((df['title'] + " " + df['abstract']))#.str.lower()

    def _extract_yake(self, docs):
        gabung_text = " ".join(docs)

        extractor = yake.KeywordExtractor(
            lan="en",
            n=self.n_gram,
            dedupLim=self.deduplim,
            # dedupFunc='seqm',
            # windowsSize=1,
            top=self.n_keywords * self.yake_multiplier
        )

        keywords = extractor.extract_keywords(gabung_text)
        # frasa = list({kw[0].lower() for kw in keywords})
        frasa = [kw[0].lower() for kw in keywords]

        
        if self.ngram_mode == "exact":
            frasa = [
                p for p in frasa 
                if len(p.split()) == self.n_gram
            ]
        elif self.ngram_mode == "mixed":
            # default YAKE → tidak perlu filter
            pass
        if len(frasa) < self.n_keywords:  print('!!! FRASA KURANG !!!') ; #frasa = [kw[0].lower() for kw in keywords]     # Supaya tetap stabil jumlah kandidat setelah filter: 

        
        if self.filter_generic_phrases:
            frasa = self._filter_generic_phrases(frasa)          # stopword removal

        frasa = [self._simple_plural_fix(p) for p in frasa]      # plural removal
        frasa = list(dict.fromkeys(frasa))                       # deduplicate
            
        frasa = frasa[:self.n_keywords]

        # FALLBACK jika terlalu sedikit
        # final_phrases = phrases[:self.n_keywords] if len(final_phrases) < self.n_keywords else final_phrases

        return frasa, gabung_text, keywords


    # ~~~~~~~~~~~~~~~~~~~~
    # BUILD GRAPH
    # ~~~~~~~~~~~~~~~~~~~~

    # DOCUMENT CO-OCCURENCE GRAPH v1.0
    # def _build_phrase_graph(self, docs, candidates, cluster_id):
    #     co_occurrence = defaultdict(int)
    #     G = nx.Graph()

    #     for doc in docs:
    #         doc = doc.lower()
    #         present_phrases = [phrase for phrase in candidates if phrase in doc]
    #         # present_phrases = [phrase for phrase in candidates if re.search(re.escape(phrase), doc, re.IGNORECASE)]

    #         for p1, p2 in itertools.combinations(present_phrases, 2):
    #             co_occurrence[(p1, p2)] += 1

    #         if self.debug:
    #             self.debug_info[cluster_id]["present_phrases_per_doc"].append(
    #                 present_phrases
    #             )

    #     for (p1, p2), weight in co_occurrence.items():
    #         G.add_edge(p1, p2, weight=weight)

    #     return G

    # # DOCUMENT CO-OCCURENCE GRAPH v2.0
    # def _build_phrase_graph(self, docs, candidates, cluster_id):
    #     """
    #     Robust phrase-level graph construction:
    #     - Semua kandidat ditambahkan sebagai node
    #     - Phrase matching menggunakan regex word boundary
    #     - Case insensitive
    #     - Co-occurrence dihitung per dokumen
    #     """
    
    #     co_occurrence = defaultdict(int)
    #     G = nx.Graph()
    #     G.add_nodes_from(candidates)
    
    #     phrase_patterns = {
    #         phrase: re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
    #         for phrase in candidates
    #     }
    
    #     for doc in docs:
    #         doc = doc.lower()
    
    #         present_phrases = [
    #             phrase
    #             for phrase, pattern in phrase_patterns.items()
    #             if pattern.search(doc)
    #         ]
    
    #         # Save DEbuggg
    #         if self.debug:
    #             self.debug_info[cluster_id]["present_phrases_per_doc"].append(
    #                 present_phrases
    #             )
    
    #         for p1, p2 in itertools.combinations(sorted(present_phrases), 2):
    #             co_occurrence[(p1, p2)] += 1
    
    #     for (p1, p2), weight in co_occurrence.items():
    #         G.add_edge(p1, p2, weight=weight)
    
    #     return G


    # # SLIDING-WINDOW CO-OCCURENCE GRAPH
    # def _build_phrase_graph(self, docs, candidates, cluster_id, window_size=20):
    #     """
    #     Build sliding-window phrase co-occurrence graph.
    
    #     Dua phrase dianggap co-occur jika jaraknya <= window_size token.
    #     """
    
    #     co_occurrence = defaultdict(int)
    #     G = nx.Graph()
    
    #     for doc in docs:
    #         doc_lower = doc.lower()
    #         tokens = doc_lower.split()
    
    #         # Simpan posisi kemunculan setiap phrase
    #         phrase_positions = defaultdict(list)
    
    #         for phrase in candidates:
    #             phrase_tokens = phrase.split()
    #             phrase_len = len(phrase_tokens)
    
    #             for i in range(len(tokens) - phrase_len + 1):
    #                 if tokens[i:i+phrase_len] == phrase_tokens:
    #                     phrase_positions[phrase].append(i)
    
    #         phrases = list(phrase_positions.keys())
    
    #         # Hitung jarak antar phrase
    #         for p1, p2 in itertools.combinations(phrases, 2):
    #             for pos1 in phrase_positions[p1]:
    #                 for pos2 in phrase_positions[p2]:
    #                     if abs(pos1 - pos2) <= window_size:
    #                         co_occurrence[(p1, p2)] += 1
    
    #         if self.debug:
    #             self.debug_info[cluster_id]["present_phrases_per_doc"].append(
    #                 phrases
    #             )
    
    #     # Build graph
    #     for (p1, p2), weight in co_occurrence.items():
    #         G.add_edge(p1, p2, weight=weight)
    
    #     return G


    # # Similarity GRAPH - Cosine TF-IDF
    # def _build_phrase_graph(self, docs, candidates, cluster_id, sim_threshold=0.3):
    #     """
    #     Build phrase similarity graph using TF-IDF cosine similarity.    
    #     Edge dibuat jika cosine similarity >= sim_threshold
    #     """
    #     from sklearn.feature_extraction.text import TfidfVectorizer
    #     from sklearn.metrics.pairwise import cosine_similarity

    #     G = nx.Graph()
    #     if len(candidates) < 2:  return G
    
    #     # TF-IDF vectorization antar phrase
    #     vectorizer = TfidfVectorizer()
    #     tfidf_matrix = vectorizer.fit_transform(candidates)
    
    #     sim_matrix = cosine_similarity(tfidf_matrix)
    
    #     for i in range(len(candidates)):
    #         for j in range(i + 1, len(candidates)):
    #             similarity = sim_matrix[i, j]
    
    #             if similarity >= sim_threshold:
    #                 G.add_edge(
    #                     candidates[i],
    #                     candidates[j],
    #                     weight=round(float(similarity), 3)
    #                 )
    
    #     if self.debug:
    #         self.debug_info[cluster_id]["present_phrases_per_doc"] = candidates
    
    #     return G


    # Similarity GRAPH - SBERT
    def _build_phrase_graph(
        self,
        docs,
        candidates,
        cluster_id,
        model_name="all-MiniLM-L6-v2",
        sim_threshold=0.4,
        device="cpu",
        batch_size=32,
        normalize_embeddings=False
    ):
        """
        Build phrase similarity graph using SBERT embeddings.
    
        Parameters
        ----------
        model_name : str
            Nama model SBERT (default: all-MiniLM-L6-v2)
        sim_threshold : float
            Threshold cosine similarity untuk membuat edge
        device : str
            "cpu" atau "cuda"
        batch_size : int
            Batch size saat encoding
        normalize_embeddings : bool
            Jika True, embeddings dinormalisasi sebelum cosine similarity
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        G = nx.Graph()
        if len(candidates) < 2:  return G
    
        model = SentenceTransformer(model_name, device=device)
        embeddings = model.encode(
            candidates,
            batch_size=batch_size,
            show_progress_bar=False
        )
        embeddings = np.array(embeddings)
    
        # Optional normalization
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
        
        sim_matrix = cosine_similarity(embeddings)
    
        # Build graph
        n = len(candidates)
    
        for i in range(n):
            for j in range(i + 1, n):
                similarity = sim_matrix[i, j]
    
                if similarity >= sim_threshold:
                    G.add_edge(
                        candidates[i],
                        candidates[j],
                        weight=round(float(similarity), 3)
                    )
    
        if self.debug:
            self.debug_info[cluster_id]["present_phrases_per_doc"] = candidates
            self.debug_info[cluster_id]["similarity_matrix"] = sim_matrix
    
        return G


    # ~~~~~~~~~~~~~~~~~~~~
    # END OF BUILD GRAPH
    # ~~~~~~~~~~~~~~~~~~~~

    def _textrank(
        self,
        graph,
        alpha=0.85,
        max_iter=100,
        tol=1e-6,
        weight='weight',
        personalization=None,
        dangling=None
    ):
        """
        Run TextRank (PageRank) on a graph.
    
        Parameters
        ----------
        graph : nx.Graph or nx.DiGraph
        alpha : float
            Damping factor (default 0.85)
        max_iter : int
            Maximum number of iterations
        tol : float
            Convergence tolerance
        weight : str or None
            Edge weight key (None for unweighted)
        personalization : dict or None
            Optional bias vector {node: value}
        dangling : dict or None
            Distribution for dangling nodes
    
        Returns
        -------
        top_phrases : list
        ranks : dict
        """
    
        ranks = nx.pagerank(
            graph,
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            weight=weight,
            personalization=personalization,
            dangling=dangling
        )
    
        ranked = sorted(
            ranks.items(),
            key=lambda x: x[1],
            reverse=True
        )
    
        top_phrases = [phrase for phrase, _ in ranked[:self.top_n]]
    
        return top_phrases, ranks

    # ~~~~~~~~~~~~~~~~~~~~
    # YAKE HELPER
    # ~~~~~~~~~~~~~~~~~~~~    
    def _filter_generic_phrases(self, phrases):
        """
        Remove phrases that contain too many domain-generic words.
        """
        filtered = []
    
        for phrase in phrases:
            tokens = phrase.split()
    
            if not tokens:
                continue
    
            generic_count = sum(
                token in self.domain_stopwords
                for token in tokens
            )
    
            # Buang jika semua kata generik
            if generic_count == len(tokens):
                continue
    
            # Buang jika lebih dari 50% kata generik
            if generic_count / len(tokens) > 0.4:
                continue
    
            filtered.append(phrase)
    
        return filtered

    # def _simple_plural_fix(self, phrase):
    #     """
    #     Melakukan normalisasi kasar pada kata jamak (plural) dengan menghapus akhiran 's'.
    #     Kelemahan kalau ada kata class, boss -> clas, bos
    #     """
    #     words = []
    #     for w in phrase.split():
    #         if w.endswith("s") and len(w) > 3:
    #             words.append(w[:-1])
    #         else:
    #             words.append(w)
    #     return " ".join(words)

    def _simple_plural_fix(self, phrase):
        """
        Melakukan normalisasi kasar pada kata jamak (plural) dengan menghapus akhiran 's'.
        Kelemahan kalau ada kata class, boss -> clas, bos
        """
        def normalize_word(w):
            w = w.lower()
    
            # rule 1: studies → study
            if w.endswith("ies") and len(w) > 4:
                return w[:-3] + "y"
    
            # rule 2: classes, boxes, wishes → class, box, wish
            elif w.endswith(("sses", "shes", "ches", "xes", "zes")) and len(w) > 4:
                return w[:-2]
    
            # rule 3: general plural → remove 's'
            elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
                return w[:-1]
    
            return w
    
        return " ".join(normalize_word(w) for w in phrase.split())
