# --- 1. Imports ---
import os
import shutil
import time
import uuid
from typing import List, Optional
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import EmailStr, BaseModel
from PIL import Image
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

# --- 2. Email Configuration ---
# IMPORTANT: YOU MUST REPLACE THESE VALUES
# For Gmail, use an "App Password". Search "Google App Password" to learn how.
EMAIL_CONFIG = {
    "SENDER_EMAIL": "lostandfoundresponse@gmail.com", # <--- REPLACE WITH YOUR GMAIL
    "SENDER_PASSWORD": "ujntmefubfgdfvij", # <--- REPLACE WITH YOUR GMAIL APP PASSWORD
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": 587,
}

# Setup for Jinja2 templates
# Ensure your 'templates' folder is in the same directory as main.py
env = Environment(loader=FileSystemLoader('templates'))
match_template = env.get_template('match_notification.html')
claim_template = env.get_template('claim_notification.html')

# --- 3. Startup Cleanup ---
print("Clearing old data...")
# Ensure 'templates' directory exists for Jinja2 loader
os.makedirs("./templates", exist_ok=True)
shutil.rmtree("./uploaded_images", ignore_errors=True)
shutil.rmtree("./qdrant_storage", ignore_errors=True)
time.sleep(1) # Give a moment for cleanup

# --- 4. Setup & Configuration ---
os.makedirs("./qdrant_storage", exist_ok=True)
os.makedirs("./uploaded_images", exist_ok=True)

print("Loading CLIP model...")
# Ensure sentence-transformers is installed: pip install sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")
print("CLIP model loaded successfully.")

# Ensure qdrant-client is installed: pip install qdrant-client
client = QdrantClient(path="./qdrant_storage")
found_collection = "found_items_collection"
lost_collection = "lost_items_collection"

# Create collections if they don't exist
client.recreate_collection(
    collection_name=found_collection,
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)
client.recreate_collection(
    collection_name=lost_collection,
    vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
)
print("Qdrant collections created for both found and lost items.")

app = FastAPI(
    title="Lost & Found AI",
    description="API for image-based lost and found item matching with email notifications.",
    version="8.1.0", # Version bump for accuracy improvement
)

# --- 5. Core ML & Helper Functions ---
def get_image_embedding(image_file):
    try:
        image_file.file.seek(0)
        image = Image.open(image_file.file).convert("RGB")
        image.thumbnail((512, 512), Image.Resampling.LANCZOS)
        embedding = model.encode(image)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def save_image(image_file: UploadFile, item_id: str):
    try:
        original_filename = image_file.filename or ""
        file_extension = Path(original_filename).suffix or ".jpg"
        new_filename = f"{item_id}{file_extension}"
        file_path = os.path.join("uploaded_images", new_filename)
        image_file.file.seek(0)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(image_file.file, buffer)
        return f"/images/{new_filename}"
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

# --- 6. Email Sending Logic (remains unchanged) ---
def send_claim_notification_email(reporting_item_payload: dict, matched_item_payload: dict, claimer_type: str):
    found_item_contact = {}
    lost_item_contact = {}

    if reporting_item_payload.get("finder_name"):
        found_item_contact = reporting_item_payload
        lost_item_contact = matched_item_payload
    elif reporting_item_payload.get("loser_name"):
        found_item_contact = matched_item_payload
        lost_item_contact = reporting_item_payload
    else:
        print("CRITICAL: Could not determine item types for email notification.")
        return

    html_content = claim_template.render(
        found_item_name=found_item_contact.get("item_name", "Unknown Found Item"),
        lost_item_name=lost_item_contact.get("item_name", "Unknown Lost Item"),
        finder_name=found_item_contact.get("finder_name", "A Finder"),
        finder_contact_email=found_item_contact.get("finder_contact_email", "Not Provided"),
        finder_phone=found_item_contact.get("finder_phone", "Not Provided"),
        loser_name=lost_item_contact.get("loser_name", "An Owner"),
        loser_contact_email=lost_item_contact.get("loser_contact_email", "Not Provided"),
        loser_phone=lost_item_contact.get("loser_phone", "Not Provided"),
    )

    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG["SENDER_EMAIL"]
    msg['Subject'] = f"Item Claimed on Lost & Found AI: {found_item_contact.get('item_name')} / {lost_item_contact.get('item_name')}"

    recipients = []
    if lost_item_contact.get("loser_contact_email"):
        recipients.append(lost_item_contact["loser_contact_email"])
    if found_item_contact.get("finder_contact_email"):
        recipients.append(found_item_contact["finder_contact_email"])

    if not recipients:
        print("WARNING: No valid recipients for claim notification email.")
        return

    msg['To'] = ", ".join(recipients)
    msg.attach(MIMEText(html_content, 'html'))

    try:
        with smtplib.SMTP(EMAIL_CONFIG["SMTP_SERVER"], EMAIL_CONFIG["SMTP_PORT"]) as server:
            server.starttls()
            server.login(EMAIL_CONFIG["SENDER_EMAIL"], EMAIL_CONFIG["SENDER_PASSWORD"])
            server.sendmail(EMAIL_CONFIG["SENDER_EMAIL"], recipients, msg.as_string())
            print(f"SUCCESS: Claim notification email sent for items '{found_item_contact.get('item_name')}' and '{lost_item_contact.get('item_name')}'")
    except Exception as e:
        print(f"CRITICAL: Email sending failed. Error: {e}")

# --- 7. API Endpoints ---

class ClaimRequest(BaseModel):
    reporting_item_id: str
    reporting_item_collection: str
    matched_item_id: str
    matched_item_collection: str

class ItemMatchRequest(BaseModel):
    item_id: str
    collection_name: str

class EmailRequest(BaseModel):
    email: EmailStr

@app.post("/api/report-found")
async def report_found_item(
    item_name: str = Form(...), user_name: str = Form(...),
    user_contact_email: EmailStr = Form(...), user_phone: str = Form(...),
    latitude: str = Form(...), longitude: str = Form(...),
    category: str = Form(...), image: UploadFile = File(...)
):
    embedding = get_image_embedding(image)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Could not process image for embedding.")

    item_id = str(uuid.uuid4())
    saved_image_path = save_image(image, item_id)
    if saved_image_path is None:
        raise HTTPException(status_code=500, detail="Could not save image.")

    found_item_payload = {
        "item_id": item_id,
        "item_name": item_name.strip(),
        "image_url": saved_image_path,
        "category": category,
        "finder_name": user_name,
        "finder_contact_email": user_contact_email,
        "finder_phone": user_phone,
        "upload_time": time.strftime("%Y-%m-%d"),
        "latitude": latitude,
        "longitude": longitude,
        "status": "available",
        "matched_with_id": None,
        "matched_with_name": None,
    }

    client.upsert(collection_name=found_collection,
                  points=[models.PointStruct(id=item_id, vector=embedding.tolist(), payload=found_item_payload)])

    category_filter = models.Filter(must=[
        models.FieldCondition(key="category", match=models.MatchValue(value=category))
    ])
    results = client.search(
        collection_name=lost_collection,
        query_vector=embedding,
        query_filter=category_filter,
        limit=5,
        score_threshold=0.85 # <-- INCREASED THRESHOLD FOR ACCURACY
    )

    potential_matches = []
    for res in results:
        potential_matches.append({
            **res.payload,
            "score": res.score,
            "collection": lost_collection
        })

    return {
        "reported_item_id": item_id,
        "reported_item_collection": found_collection,
        "reported_item_payload": found_item_payload,
        "potential_matches": potential_matches
    }


@app.post("/api/report-lost")
async def report_lost_item(
    item_name: str = Form(...), user_name: str = Form(...),
    user_contact_email: EmailStr = Form(...), user_phone: str = Form(...),
    latitude: str = Form(...), longitude: str = Form(...),
    category: str = Form(...), image: UploadFile = File(...)
):
    embedding = get_image_embedding(image)
    if embedding is None:
        raise HTTPException(status_code=400, detail="Could not process image for embedding.")

    item_id = str(uuid.uuid4())
    saved_image_path = save_image(image, item_id)
    if saved_image_path is None:
        raise HTTPException(status_code=500, detail="Could not save image.")

    lost_item_payload = {
        "item_id": item_id,
        "item_name": item_name.strip(),
        "image_url": saved_image_path,
        "category": category,
        "loser_name": user_name,
        "loser_contact_email": user_contact_email,
        "loser_phone": user_phone,
        "upload_time": time.strftime("%Y-%m-%d"),
        "latitude": latitude,
        "longitude": longitude,
        "status": "available",
        "matched_with_id": None,
        "matched_with_name": None,
    }

    client.upsert(collection_name=lost_collection,
                  points=[models.PointStruct(id=item_id, vector=embedding.tolist(), payload=lost_item_payload)])

    category_filter = models.Filter(must=[
        models.FieldCondition(key="category", match=models.MatchValue(value=category))
    ])
    results = client.search(
        collection_name=found_collection,
        query_vector=embedding,
        query_filter=category_filter,
        limit=5,
        score_threshold=0.85 # <-- INCREASED THRESHOLD FOR ACCURACY
    )

    potential_matches = []
    for res in results:
        potential_matches.append({
            **res.payload,
            "score": res.score,
            "collection": found_collection
        })

    return {
        "reported_item_id": item_id,
        "reported_item_collection": lost_collection,
        "reported_item_payload": lost_item_payload,
        "potential_matches": potential_matches
    }


@app.post("/api/claim-item")
# This function remains unchanged
async def claim_item(request: ClaimRequest):
    try:
        reporting_item_point = client.retrieve(collection_name=request.reporting_item_collection, ids=[request.reporting_item_id], with_payload=True)
        matched_item_point = client.retrieve(collection_name=request.matched_item_collection, ids=[request.matched_item_id], with_payload=True)

        if not reporting_item_point or not matched_item_point:
            raise HTTPException(status_code=404, detail="One or both items not found.")

        reporting_item_payload = reporting_item_point[0].payload
        matched_item_payload = matched_item_point[0].payload

        if reporting_item_payload.get("status") != "available" or matched_item_payload.get("status") != "available":
            raise HTTPException(status_code=400, detail="One or both items are not available to be claimed.")

        reporting_item_payload["status"] = "claimed"
        reporting_item_payload["matched_with_id"] = request.matched_item_id
        reporting_item_payload["matched_with_name"] = matched_item_payload.get("item_name", "Unknown Item")
        client.set_payload(collection_name=request.reporting_item_collection, payload=reporting_item_payload, points=[request.reporting_item_id])

        matched_item_payload["status"] = "claimed"
        matched_item_payload["matched_with_id"] = request.reporting_item_id
        matched_item_payload["matched_with_name"] = reporting_item_payload.get("item_name", "Unknown Item")
        client.set_payload(collection_name=request.matched_item_collection, payload=matched_item_payload, points=[request.matched_item_id])

        claimer_type = "finder" if request.reporting_item_collection == found_collection else "loser"
        send_claim_notification_email(reporting_item_payload, matched_item_payload, claimer_type)

        return {"message": "Items successfully claimed and owners notified!", "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to claim item due to server error: {e}")


@app.post("/api/mark-as-returned/{collection_name}/{item_id}")
# This function remains unchanged
async def mark_as_returned(collection_name: str, item_id: str):
    if collection_name not in [found_collection, lost_collection]:
        raise HTTPException(status_code=400, detail="Invalid collection name.")
    
    try:
        item_points = client.retrieve(collection_name=collection_name, ids=[item_id], with_payload=True)
        if not item_points:
            raise HTTPException(status_code=404, detail="Item not found.")
        
        item_payload = item_points[0].payload
        
        if item_payload.get("status") != "claimed":
            raise HTTPException(status_code=400, detail="Only claimed items can be marked as returned.")
            
        item_payload["status"] = "returned"
        client.set_payload(collection_name=collection_name, payload=item_payload, points=[item_id])
        
        matched_id = item_payload.get("matched_with_id")
        if matched_id:
            matched_collection = lost_collection if collection_name == found_collection else found_collection
            matched_points = client.retrieve(collection_name=matched_collection, ids=[matched_id], with_payload=True)
            if matched_points:
                matched_payload = matched_points[0].payload
                matched_payload["status"] = "returned"
                client.set_payload(collection_name=matched_collection, payload=matched_payload, points=[matched_id])

        return {"message": "Item and its match have been successfully marked as returned.", "status": "success"}

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {e}")


@app.post("/api/get-potential-matches")
async def get_potential_matches_for_item(request: ItemMatchRequest):
    try:
        item_point = client.retrieve(collection_name=request.collection_name, ids=[request.item_id], with_payload=True, with_vectors=True)
        if not item_point:
            raise HTTPException(status_code=404, detail="Item not found.")

        item_payload = item_point[0].payload
        item_embedding = item_point[0].vector
        item_category = item_payload.get("category")

        target_collection = lost_collection if request.collection_name == found_collection else found_collection
        
        category_filter = models.Filter(must=[
            models.FieldCondition(key="category", match=models.MatchValue(value=item_category))
        ])

        results = client.search(
            collection_name=target_collection,
            query_vector=item_embedding,
            query_filter=category_filter,
            limit=5,
            score_threshold=0.85 # <-- INCREASED THRESHOLD FOR ACCURACY
        )

        potential_matches = []
        for res in results:
            potential_matches.append({
                **res.payload,
                "score": res.score,
                "collection": target_collection
            })

        return {
            "reported_item_id": request.item_id,
            "reported_item_collection": request.collection_name,
            "reported_item_payload": item_payload,
            "potential_matches": potential_matches
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get potential matches due to server error: {e}")


@app.post("/api/get-items-by-email")
# This function remains unchanged
async def get_items_by_email(request: EmailRequest):
    email = request.email
    all_items = []
    
    found_filter = models.Filter(must=[
        models.FieldCondition(key="finder_contact_email", match=models.MatchValue(value=email))
    ])
    found_results, _ = client.scroll(collection_name=found_collection, scroll_filter=found_filter, limit=50, with_payload=True)
    for record in found_results:
        item = record.payload
        item["collection"] = found_collection
        all_items.append(item)

    lost_filter = models.Filter(must=[
        models.FieldCondition(key="loser_contact_email", match=models.MatchValue(value=email))
    ])
    lost_results, _ = client.scroll(collection_name=lost_collection, scroll_filter=lost_filter, limit=50, with_payload=True)
    for record in lost_results:
        item = record.payload
        item["collection"] = lost_collection
        all_items.append(item)
        
    return sorted(all_items, key=lambda x: x.get('upload_time'), reverse=True)


@app.get("/api/get-found-items")
# This function remains unchanged
async def get_found_items():
    response, _ = client.scroll(
        collection_name=found_collection, 
        limit=12, 
        with_payload=True, 
        order_by=models.OrderBy(key="upload_time", direction="desc")
    )
    return [record.payload for record in response]

@app.get("/api/get-lost-items")
# This function remains unchanged
async def get_lost_items():
    response, _ = client.scroll(
        collection_name=lost_collection, 
        limit=12, 
        with_payload=True, 
        order_by=models.OrderBy(key="upload_time", direction="desc")
    )
    return [record.payload for record in response]


@app.get("/api/item-details/{collection_name}/{item_id}")
# This function remains unchanged
async def get_item_details(collection_name: str, item_id: str):
    if collection_name not in [found_collection, lost_collection]:
        raise HTTPException(status_code=400, detail="Invalid collection name")
    try:
        point = client.retrieve(collection_name=collection_name, ids=[item_id], with_payload=True)
        if not point:
            raise HTTPException(status_code=404, detail="Item not found")
        return point[0].payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving item: {e}")


# --- 8. Static File Serving (DEFINED LAST) ---
app.mount("/images", StaticFiles(directory="uploaded_images"), name="images")
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# --- 9. Run the Server ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)