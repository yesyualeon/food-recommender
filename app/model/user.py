from email.policy import default
from enum import unique
from app import db, model
from datetime import datetime
from app.model.akun import Akun
from app.model.makanan import Makanan


class User(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    id_user = db.Column(db.BigInteger, db.ForeignKey(Akun.id, ondelete='CASCADE'))
    umur = db.Column(db.Integer, nullable=False)
    jenis_kelamin = db.Column(db.String(2), nullable=False)
    berat_badan = db.Column(db.Float, nullable=False)
    tinggi_badan = db.Column(db.Float, nullable=False)
    aktivitas_fisik = db.Column(db.Integer, nullable=False)
    riwayat_penyakit = db.Column(db.String(20), nullable=False)
    c_makanan = db.Column(db.String(250), nullable=False)
    pref_bahan = db.Column(db.String(250), nullable=False)
    bmr = db.Column(db.Float, nullable=False)
    jumlah_kalori_per_hari = db.Column(db.Float, nullable=False)
    sisa_kalori = db.Column(db.Float, nullable=False)


    def __repr__(self):
        return '<User {}>'.format(self.name)