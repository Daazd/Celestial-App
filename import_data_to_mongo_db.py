import pandas as pd
import pymongo
from datetime import datetime

MONGO_URI = "mongodb://localhost/celestial_db"
DB_NAME = "celestial_db"
COLLECTION_NAME = "celestialbodies"

def connect_to_mongodb():
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    return collection

def import_csv_to_mongodb(csv_file):
    df = pd.read_csv(csv_file)
    collection = connect_to_mongodb()
    
    updated_count = 0
    inserted_count = 0
    
    for _, row in df.iterrows():
        entry = row.to_dict()
        
        # Ensure all expected fields are present
        expected_fields = ['name', 'description', 'keywords', 'image_url', 'date']
        for field in expected_fields:
            if field not in entry:
                entry[field] = None
        
        existing_entry = collection.find_one({"name": entry['name']})
        
        if existing_entry:
            collection.update_one({"_id": existing_entry['_id']}, {"$set": entry})
            updated_count += 1
        else:
            result = collection.insert_one(entry)
            inserted_count += 1
    
    print(f"Updated {updated_count} entries")
    print(f"Inserted {inserted_count} new entries")
    
    verify_data(collection)

def verify_data(collection):
    print("\nVerifying data in MongoDB:")
    sample_entries = list(collection.aggregate([{"$sample": {"size": 5}}]))
    for entry in sample_entries:
        print(f"Sample entry: {entry['name']}")
        print(f"Image URL: {entry.get('image_url', 'Not found')}")
        print(f"Date: {entry.get('date', 'Not found')}")
        print(f"Keywords: {entry.get('keywords', 'Not found')}")
        print("---")
    
    total_count = collection.count_documents({})
    with_image_url = collection.count_documents({"image_url": {"$ne": None}})
    print(f"\nTotal entries in MongoDB: {total_count}")
    print(f"Entries with image URLs: {with_image_url}")

if __name__ == "__main__":
    csv_file = 'celestial_bodies.csv'
    import_csv_to_mongodb(csv_file)
    print("Import completed.")