import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import re


class TextFileLoader:
    def __init__(self, directory):
        self.directory = directory

    def load_files_as_string(self):
        all_text = []
        for filename in os.listdir(self.directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.directory, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    # Lecture du fichier et suppression des retours à la ligne
                    file_content = file.read().replace("\n", " ").strip()
                    cleaned_content = re.sub(r'[^a-zA-Z0-9 ]', '', file_content)
                    all_text.append(cleaned_content)
        return " ".join(all_text)


class Indexer:
    def __init__(self, text:str = '', chunk_size: int = 50):
        self.text = text
        self.chunk_size = chunk_size
        self.chunks = []  # Liste pour stocker les chunks et leur position
        self.embeddings = []  # Liste pour stocker les embeddings
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Modèle d'embeddings

    @staticmethod
    def preprocess(text: str) -> str:
        """Nettoie le texte en supprimant les caractères spéciaux et en le mettant en minuscules."""
        return re.sub(r'\W+', ' ', text.lower()).strip()

    def index_text(self):
        """Découpe le texte en chunks, génère des embeddings et stocke les informations."""
        processed_text = self.preprocess(self.text)
        words = processed_text.split()

        # Découper le texte en chunks
        for i in range(0, len(words), self.chunk_size):
            chunk = ' '.join(words[i:i + self.chunk_size])
            position = i
            embedding = self.model.encode(chunk)  # Générer l'embedding pour le chunk
            self.chunks.append((chunk, position))
            self.embeddings.append(embedding)

    def save_to_json(self, filename: str):
        """Sauvegarde les chunks et leurs embeddings dans un fichier JSON."""
        data = {
            'chunks': [chunk[0] for chunk in self.chunks],
            'positions': [chunk[1] for chunk in self.chunks],
            'embeddings': [embedding.tolist() for embedding in self.embeddings]  # Convertir les embeddings en liste
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    def load_from_json(self, filename: str):
        """Charge les chunks et leurs embeddings depuis un fichier JSON."""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.chunks = list(zip(data['chunks'], data['positions']))
            self.embeddings = [np.array(embedding) for embedding in data['embeddings']]  # Reconvertir en tableau numpy

    def retrieve_chunks(self, query: str):
        """Compare la requête avec les embeddings stockés pour récupérer les chunks pertinents."""
        query_embedding = self.model.encode(query)
        similarities = []

        # Calculer la similarité cosinus entre la requête et les embeddings
        for embedding in self.embeddings:
            cosine_similarity = np.dot(embedding, query_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(query_embedding))
            similarities.append(cosine_similarity)

        # Récupérer les chunks avec les plus hauts scores de similarité
        threshold = 0.5  # Seuil de similarité pour récupérer des chunks
        relevant_chunks = [self.chunks[i] for i in range(len(similarities)) if similarities[i] >= threshold]
        return relevant_chunks

    def enrich_context(self, chunks):
        """Utilise les positions des chunks pour enrichir le contexte."""
        enriched_chunks = []
        for chunk, position in chunks:
            # Ajouter le contexte en récupérant des chunks adjacents
            start = max(0, position - self.chunk_size)
            end = position + self.chunk_size
            context_chunk = ' '.join([c[0] for c in self.chunks[start:end]])  # Récupérer le texte des chunks adjacents
            enriched_chunks.append(context_chunk)
        return enriched_chunks


# Exemple d'utilisation
if __name__ == "__main__":
    directory_path = "/data/src"
    loader = TextFileLoader(directory_path)

    indexer = Indexer(loader.load_files_as_string(), chunk_size=20)
    indexer.index_text()

    # Sauvegarder les chunks dans un fichier JSON
    indexer.save_to_json("indexer_data.json")
