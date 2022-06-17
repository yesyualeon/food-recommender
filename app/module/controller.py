from crypt import methods
from turtle import heading
from unicodedata import name
from flask import render_template, request, session, redirect, url_for, make_response
from sklearn import datasets
from app import app
from app.model.akun import Akun
from app.module import UserController, AkunController, makanancontroller
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import csv
import re
import random
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
#from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
#nltk.download('stopwords')

#@app.route('/')
#def index():
#    pageData = {
#        "breadcrumb": "Dashboard",
#        "pageHeader": "Dashboard",
#        "pages": "dashboard.html"
#    }
#    return render_template("home.html", pageData=pageData)

db = SQLAlchemy(app)

@app.route("/home")
def data(): #fungsi yang akan dijalankan ketike route dipanggil
    #with open('/Users/elisha/flask-project/app/module/food_dataset_seminar.csv', encoding= 'unicode_escape') as csv_file:
    #    data = csv.reader(csv_file, delimiter=',')
    #    first_line = True
    #    dataset = []
    #    for row in data:
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
    #            "langkah" : row[9]
    #            })
    #        else:
    #            first_line = False
    data = makanancontroller.readMakanan()
    print(data)
    print(type(data))
    df = data.values.tolist()
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
            "langkah": row[8]
            })
        else:
            first_line = False
    
    return render_template('data.html', menu='data', submenu='data', dataset=dataset)
    
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

    #get top 5 recommendations without the user's preference (without score = 1)
    sim_scores = sim_scores[1:7]
    for i in sim_scores:
      print(i)
    sim_index = [i[0] for i in sim_scores]
    #print(sim_index)
    #print(pref_preprocessing['nama_makanan'].iloc[sim_index])
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

@app.route('/input')
def input():
    pageData = {
        "breadcrumb": "Input",
        "pageHeader": "Input Data"
    }
    return render_template("cobainput.html", pageData=pageData)

@app.route('/recommender', methods=["GET"])
def recommender():
    #if request.method == 'GET':
    pageData = {
        "breadcrumb": "Rekomendasi",
        "pageHeader": "Sistem Rekomendasi"
    }

    with open('/Users/elisha/flask-project/app/module/food_dataset_seminar.csv', encoding= 'unicode_escape') as csv_file:
        data = csv.reader(csv_file, delimiter=',')
        first_line = True
        dataset = []
        for row in data:
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


    berat_badan = float(request.args.get("berat_badan"))
    #berat_badan = [berat_badan]
    tinggi_badan = float(request.args.get("tinggi_badan"))
    #tinggi_badan = [tinggi_badan]
    usia = int(request.args.get("usia"))
    #usia = [usia]
    jenis_kelamin = request.args.get("jk")
    #jenis_kelamin = [jenis_kelamin]
    jenis_kelamin = int(jenis_kelamin)
    tingkat_aktivitas = request.args.get("aktivitas")
    #tingkat_aktivitas = [tingkat_aktivitas]
    tingkat_aktivitas = int(tingkat_aktivitas)
    penyakit = request.args.get("penyakit")
    #penyakit = int(penyakit)
    c_makanan = request.args.get("c_makanan")
    c_makanan = str(c_makanan)
    pref_bahan = request.args.get("pref_bahan")
    pref_bahan = str(pref_bahan)
    eaten_food = request.args.get("eaten_food")
    #eaten_food = [eaten_food]
    eaten_food_mass = request.args.get("eaten_food_mass")
    #eaten_food_mass = float(eaten_food_mass)
    kalori_harian = 0
    sisa_kalori = 0

    result = {}
    result['berat_badan'] = berat_badan
    result['tinggi_badan'] = tinggi_badan
    result['usia'] = usia


    #Hitung BMR pengguna
    if (jenis_kelamin == 2): bmr_u = 10 * berat_badan + 6.25 * tinggi_badan - 5 * usia + 5
    elif (jenis_kelamin == 1): bmr_u = 10 * berat_badan + 6.25 * tinggi_badan - 5 * usia - 161

    #Hitung Kebutuhan Kalori Harian Pengguna
    if (tingkat_aktivitas == 1): kalori_harian = bmr_u * 1.2
    elif (tingkat_aktivitas == 2): kalori_harian = bmr_u * 1.375
    elif (tingkat_aktivitas == 3): kalori_harian = bmr_u * 1.55
    elif (tingkat_aktivitas == 4): kalori_harian = bmr_u * 1.725
    elif (tingkat_aktivitas == 5): kalori_harian = bmr_u * 1.9
    result['kalori_harian'] = kalori_harian

    df = pd.DataFrame(dataset)
    #Cek kalori makanan yang sudah dimakan dan Hitung sisa kalori
    eaten_food = clean_split(eaten_food)
    eaten_food_mass = clean_split(eaten_food_mass)
    c_makanan = clean_split(c_makanan)
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
    result['sisa_kalori'] = sisa_kalori
        

    #Define a new data frame to store the preferred foods. Copy the contents of df to df_user
    df_user = pd.DataFrame(dataset)
    df_u = {'berat_badan': berat_badan,
            'tinggi_badan': tinggi_badan,
            'usia': usia,
            'jenis_kelamin': jenis_kelamin,
            'aktivitas': tingkat_aktivitas,
            'penyakit': penyakit,
            'constraint_makanan': c_makanan,
            'pref_bahan': pref_bahan,
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
    #df_u = pd.DataFrame(df_u)
    #print(df_u)
    UserController.save(df_u)
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
        print(penyakit)
        if (penyakit == '0'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (penyakit == '1'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif penyakit == '2': df_user = df_user[(df_user.protein <= (75 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (penyakit == '3'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (2500 - natrium_eaten_food))]
        else: print("invalid")
        print('df_user baru woyyyyy:')
        print(df_user)
        print(type(df_user))
    else:
    #df_user = df_user.where(c_makanan not in df_user['bahan_stemmed'])
        if (penyakit == '0'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (penyakit == '1'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (penyakit == '2'): df_user = df_user[(df_user.protein <= (75 - protein_eaten_food)) & (df_user.natrium <= (3000 - natrium_eaten_food))]
        elif (penyakit == '3'): df_user = df_user[(df_user.protein <= (100 - protein_eaten_food)) & (df_user.natrium <= (2500 - natrium_eaten_food))]
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
    pref_preprocessing = preprocessing(pref_bahan, df_u, df_user)
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
        
    return render_template("recommender.html",
                            pageData=pageData,
                            berat_badan = berat_badan,
                            tinggi_badan = tinggi_badan,
                            usia = usia,
                            kalori_harian = kalori_harian,
                            sisa_kalori = sisa_kalori,
                            menu='data', submenu='data', dataset_df=dataset_df)

    #elif request.method == 'POST':
    #    return UserController.save()


@app.route('/createakun', methods=['POST'])
def admins():
    return AkunController.buatAkun()


#@app.route('/login', methods=['GET','POST'])
#def logins():
#    if request.method == 'POST':
#        return AkunController.login()
#    else:
#        pageData = {
#        "breadcrumb": "Input",
#        "pageHeader": "Input Data"
#        }
#        return render_template("login.html", pageData=pageData)

@app.route('/', methods=['GET', 'POST'])
def login():
    pageData = {
        "breadcrumb": "Input",
        "pageHeader": "Input Data"
    }

    form = AkunController.LoginForm()

    if form.validate_on_submit():
        user = Akun.query.filter_by(name=form.name.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('input'))
        return '<h1>Invalid name or password</h1>'
        #return '<h1>' + form.name.data + ' ' + form.password.data + '</h1>'
    return render_template('login.html', form=form, pageData=pageData)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    form = AkunController.RegisterForm()

    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = Akun(name=form.name.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return '<h1>New user has been created!</h1>'
        
        #return '<h1>' + form.username.data + ' ' + form.email.data + ' ' + form.password.data + '</h1>'
    
    return render_template('signup.html', form=form)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


@app.route('/user', methods=['GET'])
def users():
    if request.method == 'GET':
        return UserController.index()

@app.route('/create/<id>', methods=['GET', 'POST'])
def inputData(id):
    pageData = {
        "breadcrumb": "Rekomendasi",
        "pageHeader": "Sistem Rekomendasi"
    }
    #if request.method == 'POST':
    berat_badan = float(request.form["berat_badan"])
    #berat_badan = [berat_badan]
    tinggi_badan = float(request.form["tinggi_badan"])
    #tinggi_badan = [tinggi_badan]
    usia = int(request.form["usia"])
    #usia = [usia]
    jenis_kelamin = request.form["jk"]
    #jenis_kelamin = [jenis_kelamin]
    jenis_kelamin = int(jenis_kelamin)
    tingkat_aktivitas = request.form["aktivitas"]
    #tingkat_aktivitas = [tingkat_aktivitas]
    tingkat_aktivitas = int(tingkat_aktivitas)
    penyakit = request.form["penyakit"]
    #penyakit = int(penyakit)
    c_makanan = request.form["c_makanan"]
    c_makanan = str(c_makanan)
    pref_bahan = request.form["pref_bahan"]
    pref_bahan = str(pref_bahan)
    eaten_food = request.form["eaten_food"]
    #eaten_food = [eaten_food]
    eaten_food_mass = request.form["eaten_food_mass"]
    waktu_makan = request.form["waktu_makan"]
    data = {
        'id_user': id,
        'berat_badan': berat_badan,
        'tinggi_badan': tinggi_badan,
        'usia': usia,
        'jenis_kelamin': jenis_kelamin,
        'tingkat_aktivitas': tingkat_aktivitas,
        'penyakit': penyakit,
        'c_makanan': c_makanan,
        'pref_bahan': pref_bahan,
        'eaten_food': eaten_food,
        'eaten_food_mass': eaten_food_mass,
        'waktu_makan': waktu_makan
    }
    print(data)
    #resp = make_response(render_template('recommender.html'))
    #resp.set_cookie('waktuID', waktu_makan)
    result = UserController.sisaKalori(data)
    return render_template("recommender.html",
                            pageData=pageData,
                            berat_badan = result['berat_badan'],
                            tinggi_badan = result['tinggi_badan'],
                            usia = result['usia'],
                            kalori_harian = result['kalori_harian'],
                            sisa_kalori = result['sisa_kalori'],
                            waktu_makan = result['waktu_makan'],
                            menu='data', submenu='data', dataset_df=result['dataset_df'])
    #else:
    #    return render_template("recommender.html", id=id, pageData=pageData)

@app.route('/user/<id>', methods=['GET', 'POST'])
def userDetail(id):
    if request.method == 'GET':
        pageData = {
            "breadcrumb": "Input",
            "pageHeader": "Input Data"
        }
        getDetail = UserController.detail(id)
        return render_template("logged_in_input.html", id=id, pageData=pageData, data=getDetail)
        #return UserController.detail(id)
    else:
        pageData = {
            "breadcrumb": "Input",
            "pageHeader": "Input Data"
        }
        id_user = request.form["id_user"]
        c_makanan = request.form["c_makanan"]
        c_makanan = str(c_makanan)
        penyakit = request.form["penyakit"]
        pref_bahan = request.form["pref_bahan"]
        pref_bahan = str(pref_bahan)
        kalori_harian = request.form["kalori_harian"]
        sisa_kalori = request.form['sisa_kalori']
        eaten_food = request.form["eaten_food"]
        #eaten_food = [eaten_food]
        eaten_food_mass = request.form["eaten_food_mass"]
        data = {
            'id_user': id_user,
            'c_makanan': c_makanan,
            'penyakit': penyakit,
            'pref_bahan': pref_bahan,
            'kalori_harian': kalori_harian,
            'sisa_kalori': sisa_kalori,
            'eaten_food': eaten_food,
            'eaten_food_mass': eaten_food_mass
        }
        getDetail = UserController.updateKalori(data)
        return render_template("logged_in_input.html", id=id, pageData=pageData, data=getDetail)
        #return UserController.ubah(id)


@app.route('/createmakanan')
def makanan():
    return makanancontroller.inputmakanan()


@app.errorhandler(404)
def notfound(error):
    return render_template("404.html")

    