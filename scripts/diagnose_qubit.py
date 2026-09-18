import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.vector_store import VectorStore
from src.image_store import ImageStore
from src.embeddings import EmbeddingService
from src.image_embeddings import ImageEmbeddingService
from src.retriever import MultimodalRetriever
from src.sources import filter_evidence_to_source, same_source_name, normalize_source_name, match_source_from_question

text_store = VectorStore.load(Path('indexes/faiss.index'), Path('indexes/metadata.json')) if Path('indexes/faiss.index').exists() else None
image_store = ImageStore.load(Path('indexes/image.faiss.index'), Path('indexes/image_metadata.json')) if Path('indexes/image.faiss.index').exists() else None

print('Text store chunks:', len(text_store.chunks) if text_store else None)
print('Image store records:', len(image_store.records) if image_store else None)
if image_store:
    for r in image_store.records:
        print('Image record:', r.document_name, '|', r.source_filename, '|', r.image_id, '|', r.image_path)

retriever = MultimodalRetriever(
    text_store=text_store,
    image_store=image_store,
    text_embeddings=EmbeddingService(),
    image_embeddings=ImageEmbeddingService(),
)

q = "What is the name of the person on the Qubit certificate?"
print('\n--- Retrieval without source filter ---')
ev_all = retriever.retrieve(q, top_k=5, mode='both')
print('Found evidence count (no filter):', len(ev_all))
for e in ev_all:
    print('Evidence:', e.document_name, '|', e.modality, '| score:', e.score, '| caption len:', len(e.caption or ''))

print('\n--- Retrieval with source filter Qubit.jpg ---')
ev_qubit = retriever.retrieve(q, top_k=5, mode='both', source_filter='Qubit.jpg')
print('Found evidence count (Qubit.jpg):', len(ev_qubit))
for e in ev_qubit:
    print('Evidence:', e.document_name, '|', e.modality, '| score:', e.score)

print('\n--- Retrieval with source filter Qubit ---')
ev_qubit_stem = retriever.retrieve(q, top_k=5, mode='both', source_filter='Qubit')
print('Found evidence count (Qubit):', len(ev_qubit_stem))

print('\n--- match_source_from_question ---')
print('match from question with Qubit.jpg:', match_source_from_question(q, ['Qubit.jpg', 'java fullstack(edu skills).pdf']))
print('match from question with Qubit.pdf:', match_source_from_question(q, ['Qubit.pdf', 'java fullstack(edu skills).pdf']))
print('match from question with Qubit:', match_source_from_question(q, ['Qubit', 'java fullstack(edu skills).pdf']))
