from app import db


class Tabel(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    nama_tabel = db.Column(db.String(250), nullable=False)


    def __repr__(self):
        return '<Tabel {}>'.format(self.name)
    