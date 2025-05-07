import pyrebase
import os
from models.user import User 
from models.db import db     

firebaseConfig = {
    'apiKey': os.getenv('FIREBASE_API_KEY'),
    'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
    'databaseURL': os.getenv('FIREBASE_DATABASE_URL'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
    'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    'appId': os.getenv('FIREBASE_APP_ID'),
    'measurementId': os.getenv('FIREBASE_MEASUREMENT_ID'),
}

firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

def login(email, password):
    try:
        firebase_user = auth.sign_in_with_email_and_password(email, password)

        # ✅ Flask context içindeyken DB query yapıyoruz
        local_user = User.query.filter_by(email=email).first()

        if not local_user:
            return {"error": "User found in Firebase but not in local database"}

        return {
            "id": str(local_user.id),               # session'da kullanılacak
            "email": local_user.email,
            "role": local_user.role,
            "firebase_uid": local_user.firebase_uid,
        }
    except Exception as e:
        return {"error": str(e)}

def signup(email, password):
    try:
        user = auth.create_user_with_email_and_password(email, password)
        return user
    except Exception as e:
        return {"error": f"Error creating user: {e}"}
