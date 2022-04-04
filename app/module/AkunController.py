from array import array
from distutils.log import Log
from ensurepip import bootstrap
import imp
from app.model.akun import Akun

from app import response, app, db
from flask import request, session, render_template
from flask_jwt_extended import JWTManager
from flask_jwt_extended import *
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_required, login_user, logout_user, current_user
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
import email_validator
from werkzeug.security import generate_password_hash, check_password_hash
import werkzeug
import email

import datetime

bootstrap = Bootstrap(app)

app.config['SECRET_KEY'] = 'Thisissupposedtobesecret'
app.config['USE_SESSION_FOR_NEXT'] = True

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return Akun.query.get(int(user_id))

class LoginForm(FlaskForm):
    name = StringField('name', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')

class RegisterForm(FlaskForm):
    email = StringField('email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    name = StringField('name', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('password', validators=[InputRequired(), Length(min=8, max=80)])



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