from app import db
from app.model.akun import Akun
from app.model.makanan import Makanan


class Rekomendasi(db.Model):
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    id_user = db.Column(db.BigInteger, db.ForeignKey(Akun.id, ondelete='CASCADE'))
    id_makanan = db.Column(db.BigInteger, db.ForeignKey(Makanan.id, ondelete='CASCADE'))


    def __repr__(self):
        return '<Rekomendasi {}>'.format(self.name)