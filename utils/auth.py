import pyrebase

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

firebase=pyrebase.initialize_app(firebaseConfig)
auth=firebase.auth()

def login():
    print("Login...")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        print("Login successful!")
    except Exception as e:
        print(f"Error logging in: {e}")
        return

def signup():
    print("Sing up...")
    email = input("Enter your email: ")
    password = input("Enter your password: ")
    try:
        user = auth.create_user_with_email_and_password(email, password)
        print("User created successfully!")
    except Exception as e:
        print(f"Error creating user: {e}")
    return

ans=input("Are you a new user? [y/n] ")

if ans == 'n':
    login()
elif ans == 'y':
    signup()