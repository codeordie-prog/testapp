import os
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)


def loader_map():
        
        LOADER_MAPPING = {

            ".csv": (CSVLoader, {"encoding": "utf-8"}), #specify encoding utf-8 very important
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".enex": (EverNoteLoader, {}),
            ".epub": (UnstructuredEPubLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".odt": (UnstructuredODTLoader, {}),
            ".pdf": (PDFMinerLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
        }

        return LOADER_MAPPING


def load_multiple_documents(dir_path: str):

        try:
            documents = []
            map = loader_map()

            for file in os.listdir(dir_path):
                ext = "." + file.rsplit(".", 1)[-1]

                if ext in map :
                    
                    loader_class, loader_args = map[ext]
                    # Create a new loader instance for each file
                    loader = loader_class(os.path.join(dir_path, file), **loader_args)
                    # Append the loaded documents from the current file
                    documents.extend(loader.load())
                else:
                    # Handle unsupported extensions (optional: log a message)
                    pass

            return documents
        except Exception as e:
          pass
