from app import db


class Makanan(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    nama_makanan = db.Column(db.String(250), nullable=False)
    energi = db.Column(db.Float, nullable=False)
    protein = db.Column(db.Float, nullable=False)
    lemak = db.Column(db.Float, nullable=False)
    karbo = db.Column(db.Float, nullable=False)
    natrium = db.Column(db.Float, nullable=False)
    bahan_bahan = db.Column(db.Text, nullable=False)
    bahan_stemmed = db.Column(db.Text, nullable=False)
    langkah = db.Column(db.Text, nullable=False)
    kategori_masakan = db.Column(db.String(50), nullable=False)


    def __repr__(self):
        return '<Makanan {}>'.format(self.name)
    