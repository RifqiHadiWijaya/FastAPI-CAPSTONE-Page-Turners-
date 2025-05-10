from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load model PCA dan KMeans
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Inisialisasi FastAPI
app = FastAPI()

# Input data format
class InputData(BaseModel):
    jenis_kelamin: int       # 0 = Laki-laki, 1 = Perempuan
    pekerjaan: int           # 0-9 sesuai mapping
    tujuan_kunjungan: int    # 0-5 sesuai mapping

@app.get("/")
def read_root():
    return {"message": "Cluster"}

@app.post("/Cluster")
def predict_cluster(data: InputData):
    try:
        # ✅ VALIDASI NILAI INPUT BERDASARKAN MAPPING
        if data.jenis_kelamin not in [0, 1]:
            raise ValueError("Jenis kelamin hanya boleh 0 (Laki-laki) atau 1 (Perempuan).")
        if data.pekerjaan not in list(range(10)):
            raise ValueError("Pekerjaan hanya boleh bernilai antara 0 (Mahasiswa) sampai 9 (Dosen).")
        if data.tujuan_kunjungan not in list(range(6)):
            raise ValueError("Tujuan kunjungan hanya boleh bernilai antara 0 sampai 5.")

        # Transformasi dan prediksi
        features = np.array([[data.jenis_kelamin, data.pekerjaan, data.tujuan_kunjungan]])
        transformed = pca.transform(features)
        cluster = int(kmeans.predict(transformed)[0])

        return {
            "cluster": cluster
        }

    except Exception as e:
        return {"error": str(e)}
