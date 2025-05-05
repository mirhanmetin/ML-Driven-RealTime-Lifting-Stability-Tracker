from flask_sqlalchemy import SQLAlchemy
import uuid

db = SQLAlchemy()

class SensorData(db.Model):
    __tablename__ = 'sensor_data'

    id = db.Column(db.UUID, primary_key=True, default=uuid.uuid4)
    athlete_id = db.Column(db.UUID, db.ForeignKey('users.id'))
    raw_data = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    athlete_rel = db.relationship('User', foreign_keys=[athlete_id])

    def __repr__(self):
        return f'<SensorData {self.id}>'
