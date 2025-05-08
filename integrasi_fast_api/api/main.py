from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
from fastapi.middleware.cors import CORSMiddleware

# Load model PCA dan KMeans
with open('api/pca.pkl', 'rb') as f:
    pca = pickle.load(f)

with open('api/model.pkl', 'rb') as f:
    kmeans = pickle.load(f)

# Inisialisasi FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Gantilah "*" dengan URL asal jika sudah online
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definisikan format input (request body)
class InputData(BaseModel):
    jenis_kelamin: int  # Misalnya: 0 = Pria, 1 = Wanita
    pekerjaan: int      # Misalnya: 0 = Pelajar, 1 = PNS, 2 = Swasta
    tujuan_kunjungan: int  # Misalnya: 0 = Baca Buku, 1 = Pinjam, dst



@app.post("/predict")
def predict_cluster(data: InputData):
    try:
        # 1. Ambil input
        features = np.array([[data.jenis_kelamin, data.pekerjaan, data.tujuan_kunjungan]])
        print("Input features:", features)

        # 2. Transformasi PCA dulu
        transformed = pca.transform(features)
        print("Transformed by PCA:", transformed)

        # 3. Baru prediksi dengan hasil PCA
        cluster = int(kmeans.predict(transformed)[0])

        return {
            "cluster": cluster,
            "pca_component_1": float(transformed[0, 0]),
            "pca_component_2": float(transformed[0, 1])
        }
    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}



