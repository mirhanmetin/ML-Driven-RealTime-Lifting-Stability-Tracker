from models.db import db
import uuid

class SensorData(db.Model):
    __tablename__ = 'sensor_data'

    id = db.Column(db.UUID, primary_key=True, default=uuid.uuid4)
    athlete = db.Column(db.UUID, db.ForeignKey('users.id'))
    session = db.Column(db.UUID, db.ForeignKey('sessions.id'), nullable=False)
    raw_data = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    athlete_rel = db.relationship('User', foreign_keys=[athlete])
    session_rel = db.relationship('Sessions', foreign_keys=[session])

    def __repr__(self):
        return f'<SensorData {self.id}>'
