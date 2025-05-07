from models.db import db 
import uuid

class PerformanceMetrics(db.Model):
    __tablename__ = 'performance_metrics'

    id = db.Column(db.UUID, primary_key=True, default=uuid.uuid4)
    session = db.Column(db.UUID, db.ForeignKey('sessions.id'))
    balance_score = db.Column(db.Float, nullable=False)
    stability_score = db.Column(db.Float, nullable=False)
    injury_risk = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    session_rel = db.relationship('Sessions', foreign_keys=[session])

    def __repr__(self):
        return f'<PerformanceMetrics {self.id}>'
