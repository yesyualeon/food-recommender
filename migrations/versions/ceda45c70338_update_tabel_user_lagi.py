"""update tabel user lagi

Revision ID: ceda45c70338
Revises: 0666166cb459
Create Date: 2022-03-01 22:38:34.614834

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'ceda45c70338'
down_revision = '0666166cb459'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('user', sa.Column('jenis_kelamin', sa.String(length=2), nullable=False))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('user', 'jenis_kelamin')
    # ### end Alembic commands ###