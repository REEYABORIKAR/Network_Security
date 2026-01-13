from pymongo import MongoClient
import certifi

uri = "mongodb+srv://reeyaborikar02_db_user:Admin123@cluster0.pfmesue.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(uri, tlsCAFile=certifi.where())

db = client.test
print("Connected to MongoDB Atlas!")
