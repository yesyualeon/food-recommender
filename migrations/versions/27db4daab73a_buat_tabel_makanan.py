"""buat tabel makanan

Revision ID: 27db4daab73a
Revises: 3c9a442dd5ce
Create Date: 2022-03-01 15:54:52.881918

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '27db4daab73a'
down_revision = '3c9a442dd5ce'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('makanan',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('nama_makanan', sa.String(length=250), nullable=False),
    sa.Column('energi', sa.Float(), nullable=False),
    sa.Column('protein', sa.Float(), nullable=False),
    sa.Column('lemak', sa.Float(), nullable=False),
    sa.Column('karbo', sa.Float(), nullable=False),
    sa.Column('natrium', sa.Float(), nullable=False),
    sa.Column('bahan_bahan', sa.Text(), nullable=False),
    sa.Column('bahan_stemmed', sa.Text(), nullable=False),
    sa.Column('langkah', sa.Text(), nullable=False),
    sa.Column('kategori_masakan', sa.String(length=50), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('makanan')
    # ### end Alembic commands ###
