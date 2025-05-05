import pyrebase

firebaseConfig = {
                'apiKey': "AIzaSyApWBIMyDlaeNXJgHCG-Mb9-MDXPBTscec",
                'authDomain': "capstone-powerlifting.firebaseapp.com",
                'databaseURL': "postgresql://postgres:EzoBNoVJIZgMoVscYsURFdcNvwRAKjac@interchange.proxy.rlwy.net:57128/railway",
                'projectId': "capstone-powerlifting",
                'storageBucket': "capstone-powerlifting.firebasestorage.app",
                'messagingSenderId': "294506962236",
                'appId': "1:294506962236:web:acc09b52bdcd09a8cf1239",
                'measurementId': "G-LHD5LTDP8S"
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