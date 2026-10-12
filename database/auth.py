import pymongo
import certifi
from passlib.hash import pbkdf2_sha256
from config import Config


class AuthManager:
    def __init__(self):
        ca = certifi.where()
        self.client = pymongo.MongoClient(
            Config.MONGODB_URL,
            tlsCAFile=ca,
            serverSelectionTimeoutMS=5000,
        )
        self.db         = self.client[Config.DB_NAME]
        self.collection = self.db.Account

    def signup(self, name: str, email: str, password: str):
        """Return (True, msg) on success, (False, msg) on failure."""
        email = email.strip().lower()
        if self.collection.find_one({'email': email}):
            return False, 'Email already registered.'

        self.collection.insert_one({
            'name':     name.strip(),
            'email':    email,
            'password': pbkdf2_sha256.hash(password),
        })
        return True, 'Account created! Please login.'

    def login(self, email: str, password: str):
        """Return (True, user_dict) on success, (False, msg) on failure."""
        email = email.strip().lower()
        user  = self.collection.find_one({'email': email})
        if user and pbkdf2_sha256.verify(password, user['password']):
            return True, {'name': user['name'], 'email': user['email']}
        return False, 'Invalid email or password.'
