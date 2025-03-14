# Skip-Gram Sentence Similarity

## Deskripsi Singkat
Proyek ini bertujuan untuk mengimplementasikan model Skip-Gram menggunakan Python untuk menghitung tingkat kesamaan (similarity) antara dua kalimat. Model ini menggunakan teknik word embedding sederhana untuk menghasilkan representasi kata, yang kemudian digunakan untuk menghitung kesamaan antar kalimat.

## Instalasi Dependensi
Sebelum menjalankan kode, instal dependensi berikut:

```bash
pip install newsapi-python nltk pandas numpy tqdm newsapi-python
```

## Cara Menggunakan

### Langkah 1: Scraping Data

Siapkan API key dari [NewsAPI](https://newsapi.org/) kemudian lakukan scraping data:

```q```: keyword berita yang ingin discrape

```form_param```: berita sejak kapan (jika free user, hanya 30 hari kebelakang)

```to```: berita hingga kapan

```language```: bahasa berita

```sort_by```: diurutkan berdasarkan

```page_size```: ukuran maksimal per halaman

```python
from newsapi import NewsApiClient

YOUR_API_KEY = 'your_api_key_here'
newsapi = NewsApiClient(api_key=YOUR_API_KEY)

# Scrape berita
df_news = newsapi.get_everything(q='bitcoin', from_param='2025-02-14', to='2025-03-14', language='en', sort_by='popularity', page_size=100)
```

### Langkah 2: Preprocessing Teks

Bersihkan teks (lowecasing, lemmatization, stopwords removal, digit removal) dari dataset:

```python
from text_processor import TextProcessor

processor = TextProcessor()
df_cleaned = df_news['content'].astype(str).apply(processor.cleanse)
```

### Langkah 3: Latih Model Skip-Gram

Latih model Skip-Gram menggunakan data yang sudah dibersihkan:

```window_size```: banyak kata yang dilibatkan dalam skipgram

```embedding_dim```: banyak elemen vektor yang dikeluarkan

```epochs```: banyak iterasi model


```python
from skipgram_trainer import SkipGramTrainer

trainer = SkipGramTrainer(window_size=2, embedding_dim=20, epochs=100)
model, word2idx, history_loss = trainer.train(df_cleaned)
```

### Langkah 4: Menghitung Similaritas

Hitung similarity antar dua kalimat:

```python
from sentence_similarity import SentenceSimilarity

sentence1 = "Donald Trump delivered a speech at CPAC."
sentence2 = "A dramatic speech was given by Donald Trump at CPAC"

similarity = SentenceSimilarity.calculate_similarity(sentence1, sentence2, model, word2idx)
print(f"Cosine similarity: {similarity:.2f}")
```

## Struktur Repository

```
skipgram-sentence-similarity/
├── skipgram_trainer.py
├── news_scraper.py
├── sentence_similarity.py
├── text_processor.py
└── README.md
```

## Hasil Eksperimen

Eksperimen dilakukan dengan variasi parameter:

- Window Size: 1, 2, 3
- Embedding Dimension: 20, 50, 100

Dari eksperimen didapatkan bahwa:

- Window size yang lebih besar cenderung menghasilkan embedding yang menangkap hubungan kontekstual lebih baik tetapi memerlukan waktu lebih lama untuk dilatih.
- Embedding dimension yang lebih tinggi memberikan representasi vektor yang lebih akurat tetapi membutuhkan lebih banyak resource komputasi.

Hasil terbaik diperoleh pada kombinasi window size 2 dan embedding dimension 50, dengan keseimbangan antara performa komputasi dan kualitas representasi embedding yang optimal.

## Kontribusi

Jika ingin memberikan kontribusi, silakan fork repository ini dan kirimkan pull request.



