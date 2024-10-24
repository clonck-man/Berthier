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


def dummy_ftn():
    pass


def main():
    # Exemple d'utilisation
    directory_path = "/data/src"
    loader = TextFileLoader(directory_path)

    # Charger tous les fichiers et obtenir un seul string
    combined_text = loader.load_files_as_string()
    print("txt_loader : " + combined_text)


if __name__ == "__main__":
    main()
