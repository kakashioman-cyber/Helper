import os
import zipfile
import gdown
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
import google.generativeai as genai

# 1. Konfigurasi Halaman Web Streamlit
st.set_page_config(page_title="Smart Data Helper", page_icon="📊", layout="centered")
st.title("📊 Smart Data Helper")
st.write("Asisten AI Pintar. Tanyakan apa saja yang ingin anda ketahui!")

# 2. Muat API Key dari .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ API Key tidak ditemukan di file .env!")
    st.stop()

# Konfigurasi API Key untuk pustaka google-generativeai
genai.configure(api_key=api_key)

# 3. Kelas Embedding Kustom disesuaikan dengan model lama
class GeminiEmbeddings:
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            # 💡 Samakan dengan ingest.py memakai models/gemini-embedding-001
            response = genai.embed_content(model="models/gemini-embedding-001", content=text, task_type="retrieval_document")
            embeddings.append(response['embedding'])
        return embeddings

    def embed_query(self, text):
        # 💡 Samakan dengan ingest.py memakai models/gemini-embedding-001
        response = genai.embed_content(model="models/gemini-embedding-001", content=text, task_type="retrieval_query")
        return response['embedding']

# 4. Inisialisasi Database ChromaDB
@st.cache_resource
def init_services():
    persistent_directory = "./chroma_db"
    embedding_function = GeminiEmbeddings()

    if not os.path.exists(persistent_directory):
        with st.spinner("Sedang mengunduh database sejarah secara aman..."):
            # Mengambil URL rahasia dari Secrets Streamlit Cloud
            db_url = st.secrets["DATABASE_URL"] 

            # 💡 Menggunakan gdown untuk download anti-corrupt dari Google Drive
            gdown.download(db_url, "chroma_db.zip", quiet=False)

            # Proses ekstraksi otomatis
            with zipfile.ZipFile("chroma_db.zip", 'r') as zip_ref:
                zip_ref.extractall(".")

            os.remove("chroma_db.zip") 

    return Chroma(persist_directory=persistent_directory, embedding_function=embedding_function)

db = init_services()

# 5. Kelola Riwayat Obrolan
if "messages" not in st.session_state:
    st.session_state.messages = []

# Membuat sidebar tombol hapus chat
with st.sidebar:
    st.header("⚙️ Pengaturan Obrolan")
    st.write("Gunakan tombol di bawah untuk membersihkan memori obrolan agar kuota API Gemini menjadi hemat.")
    
    # Logika jika tombol diklik
    if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun() # Muat ulang halaman web agar chat di layar langsung bersih

# Tampilkan pesan chat yang tersimpan
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 6. Input Chat dari Pengguna
if user_query := st.chat_input("Tanyakan tentang riset industri, rancang bangun, atau teknologi SDA di sini..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Sedang menganalisis seluruh dokumen dataset..."):
            try:
                # A. Ambil dokumen relevan dari database (Mengambil 5 chunks terbaik agar pembacaan tabel lebih lengkap)
                docs = db.similarity_search(user_query, k=5)
                context = "\n\n".join([doc.page_content for doc in docs])
                
                # B. Prompt khusus RAG tentang Helper Data Multidokumen Teknis
                prompt = f"""
                Anda adalah 'Smart Data Helper', sebuah sistem AI pakar analisis data, riset terapan industri, rekayasa rancang bangun, dan penerapan teknologi keberlanjutan pemanfaatan sumber daya alam (SDA).
                
                Tugas utama Anda adalah menjawab pertanyaan pengguna secara ringkas, profesional, dan berbasis data ilmiah HANYA berdasarkan informasi (konteks) dataset yang disediakan di bawah ini.
                
                Aturan Penting:
                1. Jawablah secara objektif sesuai baris, kolom, atau penjelasan yang tertulis pada dataset.
                2. Jika informasi data yang ditanyakan pengguna tidak tertera atau tidak ditemukan di dalam konteks di bawah ini, katakan dengan sangat sopan bahwa informasi data tersebut belum terdaftar atau tidak ditemukan di dalam dokumen referensi dataset terintegrasi saat ini. 
                3. Jangan pernah mengarang data statistik atau angka numerik yang tidak ada pada konteks.

                KONTEKS DATASET TERINTEGRASI:
                {context}

                PERTANYAAN PENGGUNA:
                {user_query}

                JAWABAN:
                """

                # C. Panggil model Gemini 2.5 Flash cara lama
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(prompt)
                
                bot_response = response.text
                st.markdown(bot_response)
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan: {e}")
