from models.db import db
import uuid

class Feedback(db.Model):
    __tablename__ = 'feedback'

    id = db.Column(db.UUID, primary_key=True, default=uuid.uuid4)
    session = db.Column(db.UUID, db.ForeignKey('sessions.id'))
    feedback_text = db.Column(db.String(500), nullable=False)
    metrics_id = db.Column(db.UUID, db.ForeignKey('performance_metrics.id'))
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp())

    session_rel = db.relationship('Sessions', foreign_keys=[session])
    metrics_rel = db.relationship('PerformanceMetrics', foreign_keys=[metrics_id])

    def __repr__(self):
        return f'<Feedback {self.id}>'
