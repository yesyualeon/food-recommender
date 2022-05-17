import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
from app import response, db, app
import time
import os
import config

from app.model.makanan import Makanan

#def inputmakanan():
#    HOST = str(os.environ.get("DB_HOST"))
#    DATABASE = str(os.environ.get("DB_DATABASE"))
#    USERNAME = str(os.environ.get("DB_USERNAME"))
#    PASSWORD = str(os.environ.get("DB_PASSWORD"))

#    engine = 'mysql+pymysql://' + USERNAME + ':' + PASSWORD + '@' + HOST + '/' + DATABASE
#    data = pd.read_csv('/Users/elisha/food-recommender/app/module/food_database4.csv')
#    #data['protein'] = float(data['protein'])
#    data['protein'] = data['protein'].astype(float)
#    data['lemak'] = data['lemak'].astype(float)
#    data['karbo'] = data['karbo'].astype(float)

#    data.to_sql('makanan', engine, if_exists='append', index=False)

#    df = pd.read_sql_table('makanan', engine)
#    df


def readMakanan():
    #HOST = str(os.environ.get("DB_HOST"))
    #DATABASE = str(os.environ.get("DB_DATABASE"))
    #USERNAME = str(os.environ.get("DB_USERNAME"))
    #PASSWORD = str(os.environ.get("DB_PASSWORD"))

    #engine = 'mysql+pymysql://' + USERNAME + ':' + PASSWORD + '@' + HOST + '/' + DATABASE
    #sql_data = pd.read_sql_table('makanan', config.Config.SQLALCHEMY_DATABASE_URI)
    #sql_data = Makanan.query.all()
    #print(sql_data)
    

    df = pd.read_sql_table(
        'makanan',
        config.Config.SQLALCHEMY_DATABASE_URI,
        index_col= 'id',
        columns=['nama_makanan','energi','protein','lemak','karbo','natrium','bahan_bahan','bahan_stemmed','langkah','kategori_masakan'],
        coerce_float=True
    )

    print(df.info())
    print(df.head())
    return df
    #try:
    #    makanan = Makanan.query.all()
    #    data = formatarray(makanan)
    #    return response.success(data, "success")
    #except Exception as e:
    #    print(e)

def formatarray(datas):
    array = []
    for i in datas:
        array.append(singleObject(i))

    return array

def singleObject(data):
    data = {
        'nama_makanan' :data.nama_makanan,
        'energi' : data.energi,
        'protein' : data.protein,
        'lemak' : data.lemak,
        'karbo' : data.karbo,
        'natrium' : data.natrium,
        'bahan_bahan' : data.bahan_bahan,
        'bahan_stemmed' : data.bahan_stemmed,
        'langkah' : data.langkah,
        'kategori_masakan' : data.kategori_masakan
    }

    return data
    