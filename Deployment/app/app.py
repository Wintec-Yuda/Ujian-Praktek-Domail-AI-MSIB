'''
	Contoh Deloyment untuk Domain Data Science (DS)
	Orbit Future Academy - AI Mastery - KM Batch 3
	Tim Deployment
	2022
'''

# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from joblib import load

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]	
@app.route("/api/deteksi",methods=['POST'])
def apiDeteksi():
	# Nilai default untuk variabel input atau features (X) ke model
	harapan_lama_sekolah = 5.1
	pengeluaran_perkapita  = 3.5
	rerata_lama_sekolah = 1.4
	usia_harapan_hidup  = 0.2
	
	if request.method=='POST':
		# Set nilai untuk variabel input atau features (X) berdasarkan input dari pengguna
		harapan_lama_sekolah = float(request.form['Harapan_Lama_Sekolah'])
		pengeluaran_perkapita = float(request.form['Pengeluaran_Perkapita'])
		rerata_lama_sekolah = float(request.form['Rerata_Lama_Sekolah'])
		usia_harapan_hidup = float(request.form['Usia_Harapan_Hidup'])
		
		# Prediksi kelas atau spesies bunga iris berdasarkan data pengukuran yg diberikan pengguna
		df_test = pd.DataFrame(data={
			"Harapan_Lama_Sekolah": [harapan_lama_sekolah],
			"Pengeluaran_Perkapita": [pengeluaran_perkapita],
			"Rerata_Lama_Sekolah": [rerata_lama_sekolah],
			"Usia_Harapan_Hidup": [usia_harapan_hidup]
		})

		hasil_prediksi = model.predict(df_test[0:1])[0]

		# Set Path untuk gambar hasil prediksi
		if hasil_prediksi == 1:
			# gambar_prediksi = '/static/images/iris_setosa.jpg'
			hasil_prediksi = 'High'
		elif hasil_prediksi == 2:
			# gambar_prediksi = '/static/images/iris_versicolor.jpg'
			hasil_prediksi = 'Normal'
		elif hasil_prediksi == 3:
			# gambar_prediksi = '/static/images/iris_versicolor.jpg'
			hasil_prediksi = 'Veri-Heigh'
		else:
			# gambar_prediksi = '/static/images/iris_virginica.jpg'
			hasil_prediksi = 'Low'
		
		# Return hasil prediksi dengan format JSON
		return jsonify({
			"prediksi": hasil_prediksi,
			"gambar_prediksi": '/static/images/ipm.jpg'
		})

# =[Main]========================================

if __name__ == '__main__':
	
	# Load model yang telah ditraining
	model = load('model_ipm_rf.model')

	# Run Flask di localhost 
	app.run(host="localhost", port=5000, debug=True)