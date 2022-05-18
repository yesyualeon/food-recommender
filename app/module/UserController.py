from unittest import result
import pandas as pd
from app.model.user import User
from app.model.rekomendasi import Rekomendasi
from app.model.makanan import Makanan
from app.module import makanancontroller
import csv
from app import response, app, db
from flask import request, render_template, redirect, url_for
import numpy as np
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-z #+_]')
stopworda = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower() # lowercase text
    #text = clean_spcl.sub(' ', text)
    #text = clean_symbol.sub('', text)
    #text = [item.strip() for item in text.split(",")]
    text = ' '.join(word for word in text.split() if word not in stopworda) # hapus stopword
    return text

def clean_split(text):
  text = text.strip()
  text = text.lower().split(', ')
  return text


def preprocessing(pref_bahan, df_u, df_user):
    #Preprocessing preferensi bahan
    #pref_bahan_clean = clean_text_series(df_u['pref_bahan'])
    input = pref_bahan
    clean = [item.strip() for item in input.split(",")]
    print("ini clean:",clean)
    combined = [item.replace(" ","") for item in clean]
    print("ini combined:",combined)
    print(type(combined))
    combined = ', '.join(map(lambda x: "'" + x + "'", combined))
    print("ini combined quoted:",combined)
    print(type(combined))
    # Buat kolom tambahan untuk data description yang telah dibersihkan   
    df2 = [{'nama_makanan': 'preferensi user', 'energi': 0, 'protein': 0, 'lemak': 0, 'karbo': 0, 'natrium': 0,'bahan_bahan':'', 'bahan_stemmed': combined, 'langkah':''}]
    print(df2)
    df_user = df_user.append(df2, ignore_index = True)
    print(df_user)
    return df_user


def tfidf_func(df_user):
  # PROSES TF - IDF
    #tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='indonesian')
    tf = TfidfVectorizer(analyzer='word', ngram_range=(1,1), min_df=0, binary=True)
    tfidf_matrix = tf.fit_transform(df_user['bahan_stemmed'])
    return tfidf_matrix


def cossim_func(tfidf_matrix):
    # PROSES SIMILARITY
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cos_sim


def recommendation_result(pref_preprocessing, cosine_sim = cossim_func):
  # Set nama makanan sebagai index
    #df_user.set_index('nama_makanan', inplace=True)
    indices = pd.Series(pref_preprocessing.index,index=pref_preprocessing['nama_makanan']).drop_duplicates()

    #locate nama makanan
    #idx = indices[indices == 'preferensi user'].index[0]
    idx = indices['preferensi user']

    #do tfidf
    tfidf_matrix = tfidf_func(pref_preprocessing)

    #do cosine similarity
    cos_sim = cossim_func(tfidf_matrix)

    #do cosine similarity to the user's preference
    sim_scores = enumerate(cos_sim[idx])

    #sort based on the highest similarity score
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    print("ini isi sim_scores: ")
    print(sim_scores)
    print(type(sim_scores))
    print ("Number of items in the list = ", len(sim_scores))
    print("print nonzero:", np.count_nonzero(sim_scores, axis=0))

    #get top 5 recommendations without the user's preference (without score = 1)
    sim_scores = sim_scores[1:7]
    for i in sim_scores:
      print(i)
    sim_index = [i[0] for i in sim_scores]
    print("ini isi sim_index:")
    print(sim_index)
    print("ini isi recommended_food:")
    print(pref_preprocessing['nama_makanan'].iloc[sim_index])
    #recommended_food = print(pref_preprocessing['nama_makanan'].iloc[sim_index])
    recommended_food = pref_preprocessing['nama_makanan'].iloc[sim_index]
    data = {'Nama Makanan':recommended_food,
            'Energi':pref_preprocessing['energi'].iloc[sim_index],
            'Protein':pref_preprocessing['protein'].iloc[sim_index],
            'Lemak':pref_preprocessing['lemak'].iloc[sim_index],
            'Karbo':pref_preprocessing['karbo'].iloc[sim_index],
            'Natrium':pref_preprocessing['natrium'].iloc[sim_index],
            'Bahan - bahan':pref_preprocessing['bahan_bahan'].iloc[sim_index],
            'Langkah':pref_preprocessing['langkah'].iloc[sim_index],
            'Pagi':pref_preprocessing['pagi'].iloc[sim_index],
            'siang':pref_preprocessing['siang'].iloc[sim_index],
            'malam':pref_preprocessing['malam'].iloc[sim_index]
            }
    data = pd.DataFrame(data)
    return data


def sisaKalori(data):
    pageData = {
        "breadcrumb": "Rekomendasi",
        "pageHeader": "Sistem Rekomendasi"
    }


    #eaten_food_mass = float(eaten_food_mass)
    kalori_harian = 0
    sisa_kalori = 0

    #with open('/Users/elisha/flask-project/app/module/food_dataset_seminar.csv', encoding= 'unicode_escape') as csv_file:
    #    data_apa = csv.reader(csv_file, delimiter=',')
    #    first_line = True
    #    dataset = []
    #    for row in data_apa:
    #        if not first_line:
    #            dataset.append({
    #            "nama_makanan": row[1],
    #            "energi": row[2],
    #            "protein": row[3],
    #            "lemak": row[4],
    #            "karbo":row[5],
    #            "natrium" : row[6],
    #            "bahan_bahan": row[7],
    #            "bahan_stemmed" : row[8],
    #            "langkah" : row[9],
    #            "kategori_masakan": row[10]
    #            })
    #        else:
    #            first_line = False

    data_apa = makanancontroller.readMakanan()
    print(data_apa)
    print(type(data_apa))
    df = data_apa.values.tolist()
    print(df)
    first_line = True
    dataset = []
    for row in df:
        if not first_line:
            dataset.append({
            "nama_makanan": row[0],
            "energi": row[1],
            "protein": row[2],
            "lemak": row[3],
            "karbo":row[4],
            "natrium" : row[5],
            "bahan_bahan": row[6],
            "bahan_stemmed": row[7],
            "langkah": row[8],
            "kategori_masakan": row[9]
            })
        else:
            first_line = False

    #Hitung BMR pengguna
    if (data['jenis_kelamin'] == 2): bmr_u = 10 * data['berat_badan'] + 6.25 * data['tinggi_badan'] - 5 * data['usia'] + 5
    elif (data['jenis_kelamin'] == 1): bmr_u = 10 * data['berat_badan'] + 6.25 * data['tinggi_badan'] - 5 * data['usia'] - 161

    #Hitung Kebutuhan Kalori Harian Pengguna
    if (data['tingkat_aktivitas'] == 1): kalori_harian = bmr_u * 1.2
    elif (data['tingkat_aktivitas'] == 2): kalori_harian = bmr_u * 1.375
    elif (data['tingkat_aktivitas'] == 3): kalori_harian = bmr_u * 1.55
    elif (data['tingkat_aktivitas'] == 4): kalori_harian = bmr_u * 1.725
    elif (data['tingkat_aktivitas'] == 5): kalori_harian = bmr_u * 1.9
    

    df = pd.DataFrame(dataset)
    #Cek kalori makanan yang sudah dimakan dan Hitung sisa kalori
    eaten_food = clean_split(data['eaten_food'])
    eaten_food_mass = clean_split(data['eaten_food_mass'])
    c_makanan = clean_split(data['c_makanan'])
    print(c_makanan)
    temp_kal = 0
    temp_prot = 0
    temp_nat = 0
    temp_total = kalori_harian
    print(eaten_food_mass)
    for i,st in enumerate(eaten_food):
        print(i,st)
        temp = 0
        temp_2 = 0
        temp_3 = 0
        if (st in df['nama_makanan'].values):
            for j,mass in enumerate(eaten_food_mass):
                if i == j:
                    print(st,mass)
                    mass = float(mass)
                    temp = float(df.loc[df.nama_makanan == st,'energi'])
                    #temp_kal = temp_kal + temp
                    #print(temp_kal)
                    #kalori_eaten_food = temp_kal
                    temp_total = temp_total - (temp * (mass/100))
                    print("ini sisa kalori")
                    print(temp_total)
                    temp_2 = float(df.loc[df.nama_makanan == st,'protein'])
                    temp_prot = temp_prot + (temp_2 * (mass/100))
                    print(temp_prot)
                    temp_3 = float(df.loc[df.nama_makanan == st,'natrium'])
                    temp_nat = temp_nat + (temp_3 * (mass/100))
                    print(temp_nat)
    print(eaten_food_mass)
    for mass in eaten_food_mass:
        temp = 0
        mass = float(mass)
        print(mass)
    #kalori_eaten_food = temp_kal
    protein_eaten_food = temp_prot
    natrium_eaten_food = temp_nat
    sisa_kalori = temp_total
    kalori_pagi = 25 * sisa_kalori / 100
    kalori_siang = 30 * sisa_kalori / 100
    kalori_malam = 25 * sisa_kalori / 100
    kalori_snack_pagi = 10 * sisa_kalori / 100
    kalori_snack_sore = 10 * sisa_kalori / 100

    data_user = {
            'berat_badan': data['berat_badan'],
            'tinggi_badan': data['tinggi_badan'],
            'usia': data['usia'],
            'jenis_kelamin': data['jenis_kelamin'],
            'tingkat_aktivitas': data['tingkat_aktivitas'],
            'penyakit': data['penyakit'],
            'c_makanan': c_makanan,
            'pref_bahan': data['pref_bahan'],
            'eaten_food': eaten_food,
            'eaten_food_mass': eaten_food_mass,
            'id_user':data['id_user'],
            'bmr':bmr_u,
            'kalori_harian':kalori_harian,
            'sisa_kalori': sisa_kalori
    }

    print("ini data user:", data_user)
    #result = save(data_user)
    
    ### INSERT DB ###
    save(data_user)

    #Define a new data frame to store the preferred foods. Copy the contents of df to df_user
    df_user = pd.DataFrame(dataset)
    df_u = {'berat_badan': data['berat_badan'],
            'tinggi_badan': data['tinggi_badan'],
            'usia': data['usia'],
            'jenis_kelamin': data['jenis_kelamin'],
            'aktivitas': data['tingkat_aktivitas'],
            'penyakit': data['penyakit'],
            'constraint_makanan': c_makanan,
            'pref_bahan': data['pref_bahan'],
            'eaten_food': eaten_food,
            'eaten_food_mass': eaten_food_mass,
            'bmr': bmr_u,
            'kalori_harian': kalori_harian,
            'sisa_kalori': sisa_kalori,
            'kalori_pagi': kalori_pagi,
            'kalori_siang': kalori_siang,
            'kalori_malam': kalori_malam,
            'kalori_snack_pagi': kalori_snack_pagi,
            'kalori_snack_sore': kalori_snack_sore}
    print(type(df_user))
    df_user['protein'] = df_user['protein'].astype(float)
    df_user['natrium'] = df_user['natrium'].astype(float)
    df_user['energi'] = df_user['energi'].astype(float)
    print(type(df_user))

    #c_makanan = df_u['constraint_makanan'].values[0]
    #c_makanan = str(c_makanan)
    #print(c_makanan)
    df_user_new = pd.DataFrame(dataset)

    #Filter based on the condition
    df_user = df_user[~df_user['kategori_masakan'].str.contains('side_dish')]
    df_user = df_user[df_user['energi'] <= sisa_kalori]
    if (c_makanan == '0'):
        print(df_user)
        print("df user sebelum masuk ke filtering:")
        print(type(df_user))
        print('penyakit:')
        print(data['penyakit'])
        if (data['penyakit'] == '0'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (data['penyakit'] == '1'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif data['penyakit'] == '2': df_user = df_user[(df_user.protein <= (75 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (data['penyakit'] == '3'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (2500 - natrium_eaten_food))]
        else: print("invalid")
        print('df_user baru woyyyyy:')
        print(df_user)
        print(type(df_user))
    else:
    #df_user = df_user.where(c_makanan not in df_user['bahan_stemmed'])
        if (data['penyakit'] == '0'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (data['penyakit'] == '1'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (data['penyakit'] == '2'): df_user = df_user[(df_user.protein <= (75 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (data['penyakit'] == '3'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (2500 - natrium_eaten_food))]
        else: print("invalid")
        for cons in c_makanan:    
            df_user = df_user[~df_user['bahan_stemmed'].str.contains(cons)]
        print('df_user baru woyyyyy:')
        print(df_user)
        print(type(df_user))
    
    print("df user hasil KB:")
    print(df_user)

    df_user['pagi'] = np.round((100 * kalori_pagi / df_user['energi'].values),1)
    print("ini porsi pagi:", df_user['pagi'])
    df_user['siang'] = np.round((100 * kalori_siang / df_user['energi'].values),1)
    print("ini porsi siang:", df_user['siang'])
    df_user['malam'] = np.round((100 * kalori_malam / df_user['energi'].values),1)
    print("ini porsi malam:", df_user['malam']) 
    pref_preprocessing = preprocessing(data['pref_bahan'], df_u, df_user)
    #pref_preprocessing

    tfidf_train = tfidf_func(df_user)
    #print(tfidf_train)

    cossim_train = cossim_func(tfidf_train)
    #print(cossim_train)

    personal_recommendations = recommendation_result(pref_preprocessing)

    data_df = personal_recommendations
    print("ini data_df:")
    print(data_df.columns.values)
    print(data_df)
    data_df = personal_recommendations.values.tolist()
    print("ini list dataframe:")
    print(data_df)
    print("tipe data_Df: ")
    print(type(data_df))
    first_line = True
    dataset_df = []
    for row in data_df:
        #row = row.tolist()
        #print(isinstance(row,list))
        print("ini type row:")
        print(type(row))
        #if (isinstance(row,list)):
        #    print(row)
        #    dataset_df.append(row)
        if not first_line:
            print(row[1])
            dataset_df.append({
            "nama_makanan": row[0],
            "energi": row[1],
            "protein": row[2],
            "lemak": row[3],
            "karbo": row[4],
            "natrium": row[5],
            "bahan_bahan": row[6],
            "langkah" : row[7],
            "pagi": row[8],
            "siang": row[9],
            "malam": row[10]
            })
        else:
            first_line = False
    
    data_user_output = {
                        'berat_badan' : data['berat_badan'],
                        'tinggi_badan' : data['tinggi_badan'],
                        'usia' : data['usia'],
                        'kalori_harian' : kalori_harian,
                        'sisa_kalori' : sisa_kalori,
                        'dataset_df' : dataset_df
    }
    id_makanan = personal_recommendations.index
    id_makanan = id_makanan.tolist()
    print("list id makanan:", id_makanan)
    print(type(id_makanan))
    for i in id_makanan:
        data_rekomendasi = {
            'id_user':data['id_user'],
            'id_makanan':i
        }
        save_rec(data_rekomendasi)
    #return render_template("recommender.html",
    #                        pageData=pageData,
    #                        berat_badan = data['berat_badan'],
    #                        tinggi_badan = data['tinggi_badan'],
    #                        usia = data['usia'],
    #                        kalori_harian = kalori_harian,
    #                        sisa_kalori = sisa_kalori,
    #                        menu='data', submenu='data', dataset_df=dataset_df)
    
    return data_user_output


        


def save(data):
    try:
        print("ini data fungsi save", data)
        berat_badan = data["berat_badan"]
        tinggi_badan = data["tinggi_badan"]
        usia = data["usia"]
        jenis_kelamin = data["jenis_kelamin"]
        tingkat_aktivitas = data["tingkat_aktivitas"]
        penyakit = data["penyakit"]
        c_makanan = str(data["c_makanan"])
        pref_bahan = data["pref_bahan"]
        eaten_food = data["eaten_food"]
        eaten_food_mass = data["eaten_food_mass"]
        id_user = data["id_user"]
        bmr = data["bmr"]
        kalori_harian = data["kalori_harian"]
        sisa_kalori = data["sisa_kalori"]

        #tampung constructor nya
        users = User(id_user=id_user, umur=usia, jenis_kelamin=jenis_kelamin, berat_badan=berat_badan, tinggi_badan=tinggi_badan, aktivitas_fisik=tingkat_aktivitas, riwayat_penyakit=penyakit, c_makanan=c_makanan, pref_bahan=pref_bahan, bmr=bmr, jumlah_kalori_per_hari=kalori_harian, sisa_kalori=sisa_kalori)
        db.session.add(users)
        db.session.commit()

        return response.success('', 'Sukses menambahkan data user!')
    except Exception as e:
        print(e)


def save_rec(data):
    try:
        print("ini data fungsi save", data)
        id_user = data["id_user"]
        id_makanan = data["id_makanan"]

        #tampung constructor nya
        foods = Rekomendasi(id_user=id_user, id_makanan=id_makanan)
        db.session.add(foods)
        db.session.commit()

        return response.success('', 'Sukses menambahkan data user!')
    except Exception as e:
        print(e)


#get all user
def index():
    try:
        #select all
        users = User.query.all()
        #bikin format
        data = formatarray(users)
        return response.success(data, "success")
    except Exception as e:
        print(e)

# append to array
def formatarray(datas):
    array = []

    for i in datas:
        array.append(singleObject(i))

    return array

#fungsi format dalam bentuk single object
def singleObject(data):
    data = {
        'id': data.id,
        'usia': data.umur,
        'berat_badan': data.berat_badan,
        'tinggi_badan': data.tinggi_badan,
        'aktivitas_fisik': data.aktivitas_fisik,
        'riwayat_penyakit': data.riwayat_penyakit,
        'pref_bahan': data.pref_bahan,
        'bmr': data.bmr,
        'kalori_harian': data.jumlah_kalori_per_hari,
        'sisa_kalori': data.sisa_kalori,
        'id_user': data.id_user,
        'jenis_kelamin': data.jenis_kelamin,
        'c_makanan': data.c_makanan
    }
    
    return data


#get detail user
def detail(id_user):
    try:
        data_user = User.query.filter_by(id_user=id_user).first()
        if not data_user:
            return response.badRequest([], 'Tidak ada data user!')
        #data_users = formatUser(data_user)
        data_users = singleUser(data_user)
        #return response.success(data_users, "success")
        return data_users
    except Exception as e:
        print(e)

def singleUser(data_user):
    data = {
        'id': data_user.id,
        'usia': data_user.umur,
        'berat_badan': data_user.berat_badan,
        'tinggi_badan': data_user.tinggi_badan,
        'aktivitas_fisik': data_user.aktivitas_fisik,
        'riwayat_penyakit': data_user.riwayat_penyakit,
        'pref_bahan': data_user.pref_bahan,
        'bmr': data_user.bmr,
        'kalori_harian': data_user.jumlah_kalori_per_hari,
        'sisa_kalori': data_user.sisa_kalori,
        'id_user': data_user.id_user,
        'jenis_kelamin': data_user.jenis_kelamin,
        'c_makanan': data_user.c_makanan
    }

    return data

def formatUser(data):
    array = []
    for i in data:
        array.append(singleUser(i))
    return array

#update data pref bahan
def updateKalori(data):
#eaten_food_mass = float(eaten_food_mass)
    kalori_harian = float(data['kalori_harian'])
    sisa_kalori = float(data['sisa_kalori'])
    print(sisa_kalori)
    print(kalori_harian)
    print(type(kalori_harian))

    with open('/Users/elisha/flask-project/app/module/food_dataset_seminar.csv', encoding= 'unicode_escape') as csv_file:
        data_apa = csv.reader(csv_file, delimiter=',')
        first_line = True
        dataset = []
        for row in data_apa:
            if not first_line:
                dataset.append({
                "nama_makanan": row[1],
                "energi": row[2],
                "protein": row[3],
                "lemak": row[4],
                "karbo":row[5],
                "natrium" : row[6],
                "bahan_bahan": row[7],
                "bahan_stemmed" : row[8],
                "langkah" : row[9],
                "kategori_masakan": row[10]
                })
            else:
                first_line = False

    df = pd.DataFrame(dataset)
    #Cek kalori makanan yang sudah dimakan dan Hitung sisa kalori
    eaten_food = clean_split(data['eaten_food'])
    eaten_food_mass = clean_split(data['eaten_food_mass'])
    c_makanan = clean_split(data['c_makanan'])
    print(c_makanan)
    temp_kal = 0
    temp_prot = 0
    temp_nat = 0
    temp_total = sisa_kalori
    print("ini kalori harian sebelum dihitung lagi",temp_total)
    print(eaten_food_mass)
    for i,st in enumerate(eaten_food):
        print(i,st)
        temp = 0
        temp_2 = 0
        temp_3 = 0
        if (st in df['nama_makanan'].values):
            for j,mass in enumerate(eaten_food_mass):
                if i == j:
                    print(st,mass)
                    mass = float(mass)
                    temp = float(df.loc[df.nama_makanan == st,'energi'])
                    print(temp)
                    #temp_kal = temp_kal + temp
                    #print(temp_kal)
                    #kalori_eaten_food = temp_kal
                    print(temp_total)
                    temp_total = temp_total - (temp * (mass/100))
                    print("ini sisa kalori")
                    print(temp_total)
                    temp_2 = float(df.loc[df.nama_makanan == st,'protein'])
                    temp_prot = temp_prot + (temp_2 * (mass/100))
                    print(temp_prot)
                    temp_3 = float(df.loc[df.nama_makanan == st,'natrium'])
                    temp_nat = temp_nat + (temp_3 * (mass/100))
                    print(temp_nat)
    print(eaten_food_mass)
    for mass in eaten_food_mass:
        temp = 0
        mass = float(mass)
        print(mass)
    #kalori_eaten_food = temp_kal
    protein_eaten_food = temp_prot
    natrium_eaten_food = temp_nat
    sisa_kalori = temp_total
    kalori_pagi = 25 * sisa_kalori / 100
    kalori_siang = 30 * sisa_kalori / 100
    kalori_malam = 25 * sisa_kalori / 100
    kalori_snack_pagi = 10 * sisa_kalori / 100
    kalori_snack_sore = 10 * sisa_kalori / 100
    
    print("ini sisa kalori: ",sisa_kalori)
    print("ini kalori harian: ",kalori_harian)
    data_user = {
            'id_user':data['id_user'],
            'kalori_harian':kalori_harian,
            'sisa_kalori': sisa_kalori
    }

    result = ubah(data_user)
    print('ini result eh',result)


    return result


def ubah(data):
    try:
        #eaten_food = request.form.get('eaten_food')
        #pref_bahan = request.form.get('pref_bahan')

        #input = [
        #    {
        #        'eaten_food': eaten_food,
        #        'pref_bahan': pref_bahan
        #    }
        #]

        id_user = data["id_user"]
        kalori_harian = data["kalori_harian"]
        sisa_kalori = data["sisa_kalori"]
        print(sisa_kalori)
        print("ini id_user",id_user)

        data_user = User.query.filter_by(id_user=id_user).first()
        data_apa_gitu = singleUser(data_user)
        print(data_apa_gitu)
        #data_user.eaten_food = eaten_food
        #data_user.pref_bahan = pref_bahan
        data_user.sisa_kalori = sisa_kalori
        print(data_user.sisa_kalori)

        db.session.commit()

        return 'berhasil update'
    except Exception as e:
        print(e)