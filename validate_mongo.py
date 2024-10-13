import pymongo

MONGO_URI = "mongodb://localhost/celestial_db"
DB_NAME = "celestial_db"
COLLECTION_NAME = "celestialbodies"

def verify_mongodb_data():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # Check a few random entries
    sample_entries = list(collection.aggregate([{"$sample": {"size": 5}}]))
    for entry in sample_entries:
        print(f"Name: {entry['name']}")
        print(f"Image URL: {entry.get('image_url', 'Not found')}")
        print("---")

if __name__ == "__main__":
    verify_mongodb_data()