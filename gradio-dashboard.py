import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pandas as pd
import numpy as np
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import torch

import gradio as gr

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"

books = pd.read_csv("books_with_emotions.csv")

books['large_thumbnail'] = books['thumbnail'] + '&fife=w800'
books['large_thumbnail'] = np.where(
    books['large_thumbnail'].isna(),
    'cover-not-found.jpg',
    books['large_thumbnail']
)

# embeddings = HuggingFaceEmbeddings(
#     model_name="all-MiniLM-L6-v2",
#     model_kwargs={'device': device}
# )

embeddings = OpenAIEmbeddings()

index_path = 'faiss_index_openai'

if os.path.exists(index_path):
    # Load a previously saved index:
    db_books = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    raw_documents = TextLoader('tagged_description.txt', encoding='utf-8').load()
    text_splitter = CharacterTextSplitter(chunk_size=0, chunk_overlap=0, separator="\n")
    documents = text_splitter.split_documents(raw_documents)

    db_books = FAISS.from_documents(
        documents,
        embedding=embeddings
    )
    # You can optionally save the FAISS index to disk
    db_books.save_local(index_path)
def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone:str = None,
        initial_top_k:int = 50,
        final_top_k:int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [rec.page_content.strip('"').split()[0] for rec in recs]
    book_recs = books[books['isbn10'].isin(books_list)].head(final_top_k)

    if category != 'All':
        book_recs = book_recs[book_recs['simple_categories'] == category]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs

def recommend_books(
        query: str,
        category: str,
        tone:str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row['description']
        # split description into words
        truncated_desc_split = description.split()
        # truncate description to 30 words or less
        truncated_description = ' '.join(truncated_desc_split[:60]) + '...'

        authors_str = row['authors']

        caption = f"{row['title']} by {authors_str}: {truncated_description}"

        results.append((row['large_thumbnail'], caption))

    return results

categories = ["All"] + sorted(books['simple_categories'].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown('# Semantic book recommender')

    with gr.Row():
        user_query = gr.Textbox(label="Enter a book description", placeholder="e.g., A story about love...")
        category_dropdown = gr.Dropdown(categories, label="Select a category", value="All")
        tone_dropdown = gr.Dropdown(tones, label="Select a tone", value="All")
        submit_button = gr.Button(value="Find recommendations")

    gr.Markdown("## Recommendations")

    output = gr.Gallery(label="Recommended books", rows=2, columns=8)

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output,
    )

if __name__ == "__main__":
    dashboard.launch()

