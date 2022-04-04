from array import array
from app.model.akun import Akun

from app import response, app, db
from flask import request, session, render_template
from flask_jwt_extended import JWTManager
from flask_jwt_extended import *

import datetime

def buatAkun():
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')

        akuns = Akun(name=name, email=email)
        akuns.setPassword(password)

        db.session.add(akuns)
        db.session.commit()

        return response.success('', 'Sukses menambahkan akun user!')

    except Exception as e:
        print(e)

#fungsi ambil format data untuk login
def singleObject(data):
    data = {
        'id' : data.id,
        'name': data.name,
        'email': data.email
    }

    return data

#fungsi login
def login():
    try:
        email = request.form.get('email')
        password = request.form.get('password')

        #cek user, apakah email terdaftar atu tidak
        akun = Akun.query.filter_by(email=email).first()

        if not akun:
            return response.badRequest([], 'Email tidak terdaftar!')
        
        #cek password ada atau engga
        if not akun.checkPassword(password):
            return response.badRequest([], 'Kombinasi password salah')

        #tampilkan dalam bentuk singleobject
        data = singleObject(akun)
        print(data)


        #set expired date tokennya
        expires = datetime.timedelta(days=7)
        print(expires)
        #set refresh token 
        expires_refresh = datetime.timedelta(days=7)
        print(expires_refresh)

        #akses tokennya
        access_token = create_access_token(data, fresh=True, expires_delta=expires)
        print(access_token)
        #jika akses token udah habis, diperpanjang tokennya pake refresh token
        refresh_token = create_refresh_token(data, expires_delta=expires_refresh)
        print(refresh_token)

        #buat session
        #session['name'] = akun['name']
        #session['email'] = akun['email']

        #pageData = {
        #"breadcrumb": "Input",
        #"pageHeader": "Input Data"
        #}

        return response.success({
            "data": data,
            "access_token": access_token,
            "refresh_token": refresh_token,
        }, "Sukses Login!")
        #return render_template("logged_in_input.html",
        #                        data = data,
        #                        access_token = access_token,
        #                        refresh_token = refresh_token,
        #                        pageData=pageData)
    except Exception as e:
        print(e)