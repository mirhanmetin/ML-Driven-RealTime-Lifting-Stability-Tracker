from models.db import db 
import uuid

class Sessions(db.Model):
    __tablename__ = 'sessions'

    id = db.Column(db.UUID, primary_key=True, default=uuid.uuid4)
    trainer = db.Column(db.UUID, db.ForeignKey('users.id'))
    athlete = db.Column(db.UUID, db.ForeignKey('users.id'))
    lift_type = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='ongoing')
    sensor_data_id = db.Column(db.UUID, db.ForeignKey('sensor_data.id'))
    started_at = db.Column(db.DateTime, nullable=False)
    ended_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    trainer_rel = db.relationship('User', foreign_keys=[trainer])
    athlete_rel = db.relationship('User', foreign_keys=[athlete])
    sensor_data_rel = db.relationship('SensorData', foreign_keys=[sensor_data_id])

    def __repr__(self):
        return f'<Sessions {self.id}>'
