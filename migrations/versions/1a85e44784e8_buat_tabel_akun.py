"""buat tabel akun

Revision ID: 1a85e44784e8
Revises: 27db4daab73a
Create Date: 2022-03-01 22:13:01.670946

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1a85e44784e8'
down_revision = '27db4daab73a'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('akun',
    sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
    sa.Column('name', sa.String(length=250), nullable=False),
    sa.Column('email', sa.String(length=60), nullable=False),
    sa.Column('password', sa.String(length=250), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.Column('updated_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_akun_email'), 'akun', ['email'], unique=True)
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_akun_email'), table_name='akun')
    op.drop_table('akun')
    # ### end Alembic commands ###