# --- 1. Imports ---
import os
import shutil
import time
import uuid
import io  # <<< NEW: Required for safe file reading
from typing import List, Optional
from pathlib import Path
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from jinja2 import Environment, FileSystemLoader
import re
import cv2

# --- OCR/Image Libs ---
import easyocr
from PIL import Image, ImageDraw, ImageOps, ImageEnhance

# --- Web Framework ---
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import EmailStr, BaseModel

# --- Vector DB & Embeddings ---
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np

# --- 2. Email Configuration ---
EMAIL_CONFIG = {
    "SENDER_EMAIL": "lostandfoundresponse@gmail.com",
    "SENDER_PASSWORD":"ujntmefubfgdfvij",  # <--- PASTE YOUR APP PASSWORD HERE
    "SMTP_SERVER": "smtp.gmail.com",
    "SMTP_PORT": 587,
}

# --- Jinja2 Setup ---
os.makedirs("./templates", exist_ok=True)
Path("./templates/match_notification.html").touch(exist_ok=True)
Path("./templates/claim_notification.html").touch(exist_ok=True)
env = Environment(loader=FileSystemLoader('templates'))
match_template = env.get_template('match_notification.html')
claim_template = env.get_template('claim_notification.html')

# --- 3. Startup Cleanup ---
# FIX: Commented out the destructive cleanup so you don't lose data on restart
# print(">>> Startup: Clearing old data...")
# shutil.rmtree("./uploaded_images", ignore_errors=True)
# shutil.rmtree("./qdrant_storage", ignore_errors=True)
# time.sleep(1)
# print(">>> Startup: Old data cleared.")

# --- 4. Setup & Configuration ---
os.makedirs("./qdrant_storage", exist_ok=True)
os.makedirs("./uploaded_images", exist_ok=True)

print(">>> Startup: Loading CLIP model...")
model = SentenceTransformer("clip-ViT-B-32")
print(">>> Startup: CLIP model loaded.")

print(">>> Startup: Loading EasyOCR Reader (English)... This might take a moment on first run.")
try:
    reader = easyocr.Reader(['en'], gpu=False)
    print(">>> Startup: EasyOCR Reader loaded.")
except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"ERROR: Failed to initialize EasyOCR Reader: {e}")
    print("Ensure PyTorch and EasyOCR are installed correctly.")
    print("Censorship will likely fail or be skipped.")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    reader = None

client = None
found_collection = "found_items_collection"
lost_collection = "lost_items_collection"
try:
    print(">>> Startup: Initializing QdrantClient...")
    client = QdrantClient(path="./qdrant_storage")
    print(f">>> Startup: QdrantClient initialized: {client}")

    # FIX: Safely check for existing collections instead of recreating (wiping) them
    existing_collections = [col.name for col in client.get_collections().collections]

    if found_collection not in existing_collections:
        print(f">>> Startup: Creating collection '{found_collection}'...")
        client.create_collection(
            collection_name=found_collection,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )
        print(f">>> Startup: Collection '{found_collection}' created.")
    else:
        print(f">>> Startup: Collection '{found_collection}' already exists.")

    if lost_collection not in existing_collections:
        print(f">>> Startup: Creating collection '{lost_collection}'...")
        client.create_collection(
            collection_name=lost_collection,
            vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),
        )
        print(f">>> Startup: Collection '{lost_collection}' created.")
    else:
        print(f">>> Startup: Collection '{lost_collection}' already exists.")

    print(">>> Startup: Qdrant collections ready.")

except Exception as e:
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"FATAL: Qdrant client/collection setup failed: {e}")
    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if client:
        print(">>> Startup: Client object exists, but collection creation might have failed.")
    else:
        print(">>> Startup: Qdrant client object could NOT be created.")

# --- FastAPI App ---
app = FastAPI(
    title="Lost & Found AI",
    description="API for image-based lost and found item matching with email notifications.",
    version="8.3.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 5. Core ML & Helper Functions ---

def get_image_embedding(image_pil: Image.Image):
    try:
        img_copy = image_pil.copy()
        if img_copy.mode != 'RGB':
            img_copy = img_copy.convert('RGB')
        # Removed the manual thumbnail scaling; let CLIP handle sizing internally to preserve fidelity
        embedding = model.encode(img_copy)
        if embedding is None: return None
        if isinstance(embedding, np.ndarray) and embedding.size == 0: return None
        return embedding
    except Exception as e:
        print(f"!!! CRITICAL Error during get_image_embedding: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- 6. Email Sending Logic ---
def send_claim_notification_email(reporting_item_payload: dict, matched_item_payload: dict, claimer_type: str):
    found_item_contact = {}
    lost_item_contact = {}
    if reporting_item_payload.get("finder_name") is not None:
        found_item_contact = reporting_item_payload
        lost_item_contact = matched_item_payload
    elif reporting_item_payload.get("loser_name") is not None:
        lost_item_contact = reporting_item_payload
        found_item_contact = matched_item_payload
    else:
        if matched_item_payload.get("finder_name") is not None:
            found_item_contact = matched_item_payload
            lost_item_contact = reporting_item_payload
        elif matched_item_payload.get("loser_name") is not None:
            lost_item_contact = matched_item_payload
            found_item_contact = reporting_item_payload
        else:
            print("CRITICAL: Could not determine item types for email notification.")
            return

    html_content = claim_template.render(
        found_item_name=found_item_contact.get("item_name", "Item"),
        lost_item_name=lost_item_contact.get("item_name", "Item"),
        finder_name=found_item_contact.get("finder_name", "Finder"),
        finder_contact_email=found_item_contact.get("finder_contact_email", "[Not Provided]"),
        finder_phone=found_item_contact.get("finder_phone", "[Not Provided]"),
        loser_name=lost_item_contact.get("loser_name", "Owner"),
        loser_contact_email=lost_item_contact.get("loser_contact_email", "[Not Provided]"),
        loser_phone=lost_item_contact.get("loser_phone", "[Not Provided]"),
    )
    msg = MIMEMultipart()
    msg['From'] = EMAIL_CONFIG["SENDER_EMAIL"]
    msg[
        'Subject'] = f"Item Claimed on Lost & Found AI: {found_item_contact.get('item_name', '')} / {lost_item_contact.get('item_name', '')}"

    recipients = []
    if lost_item_contact.get("loser_contact_email"): recipients.append(lost_item_contact["loser_contact_email"])
    if found_item_contact.get("finder_contact_email"): recipients.append(found_item_contact["finder_contact_email"])

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
            print(f"SUCCESS: Claim notification email sent.")
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


async def perform_easyocr_censorship(item_id: str, img_pil_to_censor: Image.Image):
    global reader
    if not reader:
        print(f"!!! ERROR: EasyOCR Reader not initialized. Skipping censorship for {item_id}.")
        return img_pil_to_censor, 0

    draw = ImageDraw.Draw(img_pil_to_censor)
    censored_count = 0
    censored_coords = []
    img_np = np.array(img_pil_to_censor)

    def is_overlapping(x0, y0, x1, y1):
        for cx0, cy0, cx1, cy1 in censored_coords:
            if x1 < cx0 or x0 > cx1 or y1 < cy0 or y0 > cy1:
                continue
            else:
                return True
        return False

    try:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        qr_detector = cv2.QRCodeDetector()
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(gray)

        if retval and points is not None:
            for pts in points:
                x_coords = [p[0] for p in pts]
                y_coords = [p[1] for p in pts]
                x0, y0 = int(min(x_coords)), int(min(y_coords))
                x1, y1 = int(max(x_coords)), int(max(y_coords))
                pad = 8
                bbox_coords = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
                draw.rectangle(bbox_coords, fill="black")
                censored_coords.append(bbox_coords)
                censored_count += 1
                print(f"[Censor] Masked QR Code via OpenCV.")
    except Exception as e:
        print(f"QR Detection Error: {e}")

    try:
        print(f"[Censor] Running EasyOCR readtext...")
        ocr_results = reader.readtext(img_np, detail=1, paragraph=False)
        pan_regex = re.compile(r'[A-Z]{5}[0-9O]{4}[A-Z1]', re.IGNORECASE)
        aadhaar_regex = re.compile(r'[0-9O]{12}', re.IGNORECASE)
        uid_regex = re.compile(r'[A-Z]{3}[0-9O]{12}', re.IGNORECASE)
        voter_id_regex = re.compile(r'[A-Z]{3}[0-9O]{7}', re.IGNORECASE)
        dl_regex = re.compile(r'[A-Z]{2}[0-9O]{13}', re.IGNORECASE)
        phone_regex = re.compile(r'(?:91)?[6-9][0-9O]{9}', re.IGNORECASE)

        for (bbox, text, conf) in ocr_results:
            original_text = text.strip()
            if not original_text or len(original_text) < 3: continue
            clean_text = re.sub(r'[\s\-]', '', original_text.upper())

            x_coords = [p[0] for p in bbox]
            y_coords = [p[1] for p in bbox]
            x0, y0 = int(min(x_coords)), int(min(y_coords))
            x1, y1 = int(max(x_coords)), int(max(y_coords))

            if is_overlapping(x0, y0, x1, y1): continue

            is_sensitive = False
            reason = ""
            if len(clean_text) > 20 and conf < 0.45:
                is_sensitive = True
                reason = "Garbled OCR Text (Likely Barcode/QR)"
            elif uid_regex.search(clean_text):
                is_sensitive = True
                reason = "UID"
            elif pan_regex.search(clean_text):
                is_sensitive = True
                reason = "PAN Card"
            elif aadhaar_regex.search(clean_text):
                is_sensitive = True
                reason = "Aadhaar Card"
            elif voter_id_regex.search(clean_text):
                is_sensitive = True
                reason = "Voter ID"
            elif dl_regex.search(clean_text):
                is_sensitive = True
                reason = "Driving License"
            elif phone_regex.search(clean_text) and len(clean_text) <= 12:
                is_sensitive = True
                reason = "Phone Number"

            if is_sensitive:
                print(f"!!! Matched '{original_text}' -> '{clean_text}' | Reason: {reason}")
                pad = 4
                bbox_coords = (x0 - pad, y0 - pad, x1 + pad, y1 + pad)
                draw.rectangle(bbox_coords, fill="black")
                censored_coords.append(bbox_coords)
                censored_count += 1
    except Exception as e:
        import traceback
        traceback.print_exc()

    del draw
    print(f"[Censor {item_id}] Auto-censored {censored_count} areas.")
    return img_pil_to_censor, censored_count


@app.post("/api/report-found")
async def report_found_item(
        item_name: str = Form(...), user_name: str = Form(...),
        user_contact_email: EmailStr = Form(...), user_phone: str = Form(...),
        latitude: str = Form(...), longitude: str = Form(...),
        category: str = Form(...), image: UploadFile = File(...)
):
    if not client: raise HTTPException(status_code=503, detail="Database client not available.")

    item_id = str(uuid.uuid4())
    original_filename = image.filename or ""
    file_extension = Path(original_filename).suffix if Path(original_filename).suffix else ".jpg"
    new_filename = f"{item_id}{file_extension}"
    save_path = os.path.join("uploaded_images", new_filename)
    image_url = f"/images/{new_filename}"

    try:
        image_data = await image.read()
        img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_for_embedding = img_pil.copy()
        img_pil_to_censor = img_pil.copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load image file: {e}")

    embedding_array = get_image_embedding(img_for_embedding)
    if embedding_array is None or not isinstance(embedding_array, np.ndarray) or embedding_array.size == 0:
        raise HTTPException(status_code=400, detail="Could not process image for embedding.")

    embedding = [float(x) for x in embedding_array.flatten().tolist()]
    clean_category = category.strip().lower()

    img_censored, count = await perform_easyocr_censorship(item_id, img_pil_to_censor)

    try:
        if img_censored.mode == 'RGBA':
            bg = Image.new('RGB', img_censored.size, (255, 255, 255))
            bg.paste(img_censored, (0, 0), img_censored)
            img_censored = bg
        img_censored.save(save_path, quality=85, optimize=True)
        print(f"Saved processed image to: {save_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save processed image: {e}")

    found_item_payload = {
        "item_id": item_id, "item_name": item_name.strip(), "image_url": image_url,
        "category": clean_category, "finder_name": user_name,
        "finder_contact_email": user_contact_email, "finder_phone": user_phone,
        "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": latitude, "longitude": longitude, "status": "available",
        "matched_with_id": None, "matched_with_name": None
    }

    try:
        print(f">>> Qdrant: Upserting {item_id} into {found_collection}...")
        client.upsert(collection_name=found_collection, points=[
            models.PointStruct(id=item_id, vector=embedding, payload=found_item_payload)], wait=True)
    except Exception as e:
        print(f"!!! ERROR Qdrant upsert: {e}")
        raise HTTPException(status_code=500, detail="Database error during item report.")

    category_filter = models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value=clean_category)),
              models.FieldCondition(key="status", match=models.MatchValue(value="available"))])

    try:
        print(f">>> Qdrant: Searching {lost_collection} for matches...")
        results = client.search(collection_name=lost_collection, query_vector=embedding, query_filter=category_filter,
                                limit=5, score_threshold=0.60)
        print(f">>> Qdrant: Search found {len(results)} matches.")
    except Exception as e:
        print(f"!!! ERROR Qdrant search: {e}")
        results = []

    potential_matches = []
    for res in results:
        match_payload = res.payload.copy()
        potential_matches.append({**match_payload, "score": res.score, "collection": lost_collection})

    return {"reported_item_id": item_id, "reported_item_collection": found_collection,
            "reported_item_payload": found_item_payload, "potential_matches": potential_matches}


@app.post("/api/report-lost")
async def report_lost_item(
        item_name: str = Form(...), user_name: str = Form(...),
        user_contact_email: EmailStr = Form(...), user_phone: str = Form(...),
        latitude: str = Form(...), longitude: str = Form(...),
        category: str = Form(...), image: UploadFile = File(...)
):
    if not client: raise HTTPException(status_code=503, detail="Database client is not available.")

    item_id = str(uuid.uuid4())
    original_filename = image.filename or ""
    file_extension = Path(original_filename).suffix if Path(original_filename).suffix else ".jpg"
    new_filename = f"{item_id}{file_extension}"
    save_path = os.path.join("uploaded_images", new_filename)
    image_url = f"/images/{new_filename}"

    try:
        image_data = await image.read()
        img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_for_embedding = img_pil.copy()
        img_pil_to_censor = img_pil.copy()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load image file: {e}")

    embedding_array = get_image_embedding(img_for_embedding)
    if embedding_array is None or not isinstance(embedding_array, np.ndarray) or embedding_array.size == 0:
        raise HTTPException(status_code=400, detail="Could not process image for embedding.")

    embedding = [float(x) for x in embedding_array.flatten().tolist()]
    clean_category = category.strip().lower()

    img_censored, count = await perform_easyocr_censorship(item_id, img_pil_to_censor)

    try:
        if img_censored.mode == 'RGBA':
            bg = Image.new('RGB', img_censored.size, (255, 255, 255))
            bg.paste(img_censored, (0, 0), img_censored)
            img_censored = bg
        img_censored.save(save_path, quality=85, optimize=True)
        print(f"Saved processed image to: {save_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save processed image: {e}")

    lost_item_payload = {
        "item_id": item_id, "item_name": item_name.strip(), "image_url": image_url,
        "category": clean_category, "loser_name": user_name, "loser_contact_email": user_contact_email,
        "loser_phone": user_phone, "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "latitude": latitude, "longitude": longitude, "status": "available",
        "matched_with_id": None, "matched_with_name": None
    }

    try:
        print(f">>> Qdrant: Upserting {item_id} into {lost_collection}...")
        client.upsert(collection_name=lost_collection, points=[
            models.PointStruct(id=item_id, vector=embedding, payload=lost_item_payload)], wait=True)
    except Exception as e:
        print(f"!!! ERROR Qdrant upsert: {e}")
        raise HTTPException(status_code=500, detail="Database error during item report.")

    category_filter = models.Filter(
        must=[models.FieldCondition(key="category", match=models.MatchValue(value=clean_category)),
              models.FieldCondition(key="status", match=models.MatchValue(value="available"))])

    try:
        print(f">>> Qdrant: Searching {found_collection} for matches...")
        results = client.search(collection_name=found_collection, query_vector=embedding, query_filter=category_filter,
                                limit=5, score_threshold=0.55)
        print(f">>> Qdrant: Search found {len(results)} matches.")
    except Exception as e:
        print(f"!!! ERROR Qdrant search: {e}")
        results = []

    potential_matches = []
    for res in results:
        match_payload = res.payload.copy()
        potential_matches.append({**match_payload, "score": res.score, "collection": found_collection})

    return {"reported_item_id": item_id, "reported_item_collection": lost_collection,
            "reported_item_payload": lost_item_payload, "potential_matches": potential_matches}


@app.post("/api/claim-item")
async def claim_item(request: ClaimRequest):
    if not client: raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        reporting_item_point = client.retrieve(collection_name=request.reporting_item_collection,
                                               ids=[request.reporting_item_id], with_payload=True)
        matched_item_point = client.retrieve(collection_name=request.matched_item_collection,
                                             ids=[request.matched_item_id], with_payload=True)
        if not reporting_item_point or not matched_item_point:
            raise HTTPException(status_code=404, detail="One or both items not found.")

        reporting_item_payload = reporting_item_point[0].payload
        matched_item_payload = matched_item_point[0].payload

        if reporting_item_payload.get("status") != "available" or matched_item_payload.get("status") != "available":
            raise HTTPException(status_code=400, detail="One or both items are not available to be claimed.")

        reporting_item_payload["status"] = "claimed"
        reporting_item_payload["matched_with_id"] = request.matched_item_id
        reporting_item_payload["matched_with_name"] = matched_item_payload.get("item_name", "Unknown Item")
        client.set_payload(collection_name=request.reporting_item_collection, payload=reporting_item_payload,
                           points=[request.reporting_item_id])

        matched_item_payload["status"] = "claimed"
        matched_item_payload["matched_with_id"] = request.reporting_item_id
        matched_item_payload["matched_with_name"] = reporting_item_payload.get("item_name", "Unknown Item")
        client.set_payload(collection_name=request.matched_item_collection, payload=matched_item_payload,
                           points=[request.matched_item_id])

        claimer_type = "finder" if request.reporting_item_collection == found_collection else "loser"
        send_claim_notification_email(reporting_item_payload, matched_item_payload, claimer_type)
        return {"message": "Items successfully claimed and owners notified!", "status": "success"}
    except Exception as e:
        print(f"Error claiming item: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to claim item due to server error.")


@app.post("/api/mark-as-returned/{collection_name}/{item_id}")
async def mark_as_returned(collection_name: str, item_id: str):
    if not client: raise HTTPException(status_code=503, detail="Database client is not available.")
    if collection_name not in [found_collection, lost_collection]:
        raise HTTPException(status_code=400, detail="Invalid collection name.")

    try:
        item_points = client.retrieve(collection_name=collection_name, ids=[item_id], with_payload=True)
        if not item_points: raise HTTPException(status_code=404, detail="Item not found.")

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
                if matched_payload.get("status") == "claimed":
                    matched_payload["status"] = "returned"
                    client.set_payload(collection_name=matched_collection, payload=matched_payload, points=[matched_id])

        return {"message": "Item (and potentially its match) marked as returned.", "status": "success"}
    except HTTPException as e:
        raise e
    except Exception as e:
        print(f"Error marking as returned: {e}")
        raise HTTPException(status_code=500, detail=f"Server error occurred.")


@app.post("/api/get-potential-matches")
async def get_potential_matches_for_item(request: ItemMatchRequest):
    if not client: raise HTTPException(status_code=503, detail="Database client is not available.")
    try:
        item_point = client.retrieve(collection_name=request.collection_name, ids=[request.item_id], with_payload=True,
                                     with_vectors=True)
        if not item_point or not item_point[0].vector:
            raise HTTPException(status_code=404, detail="Item not found or missing vector.")

        item_payload = item_point[0].payload
        item_embedding = item_point[0].vector
        item_category = item_payload.get("category", "").strip().lower()
        target_collection = lost_collection if request.collection_name == found_collection else found_collection

        category_filter = models.Filter(
            must=[models.FieldCondition(key="category", match=models.MatchValue(value=item_category)),
                  models.FieldCondition(key="status", match=models.MatchValue(value="available"))])

        results = client.search(collection_name=target_collection, query_vector=item_embedding,
                                query_filter=category_filter, limit=5, score_threshold=0.60)

        potential_matches = []
        for res in results:
            match_payload = res.payload.copy()
            potential_matches.append({**match_payload, "score": res.score, "collection": target_collection})

        return {"reported_item_id": request.item_id, "reported_item_collection": request.collection_name,
                "reported_item_payload": item_payload, "potential_matches": potential_matches}
    except Exception as e:
        print(f"Error getting potential matches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get potential matches due to server error.")


@app.post("/api/get-items-by-email")
async def get_items_by_email(request: EmailRequest):
    if not client: raise HTTPException(status_code=503, detail="Database client is not available.")
    email = request.email
    all_items = []
    try:
        found_filter = models.Filter(
            must=[models.FieldCondition(key="finder_contact_email", match=models.MatchValue(value=email))])
        found_results, _ = client.scroll(collection_name=found_collection, scroll_filter=found_filter, limit=100,
                                         with_payload=True)
        for record in found_results:
            item = record.payload
            item["collection"] = found_collection
            all_items.append(item)

        # FIX: Corrected models.Field.Condition to models.FieldCondition
        lost_filter = models.Filter(
            must=[models.FieldCondition(key="loser_contact_email", match=models.MatchValue(value=email))])
        lost_results, _ = client.scroll(collection_name=lost_collection, scroll_filter=lost_filter, limit=100,
                                        with_payload=True)
        for record in lost_results:
            item = record.payload
            item["collection"] = lost_collection
            all_items.append(item)

        return sorted(all_items, key=lambda x: x.get('upload_time', '1970-01-01 00:00:00'), reverse=True)
    except Exception as e:
        print(f"Error getting items by email: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve items.")


@app.get("/api/get-found-items")
async def get_found_items():
    if not client: return []
    try:
        response, _ = client.scroll(
            collection_name=found_collection, limit=12, with_payload=True,
            order_by=models.OrderBy(key="upload_time", direction="desc")
        )
        return [record.payload for record in response]
    except Exception as e:
        print(f"Error getting found items: {e}")
        return []


@app.get("/api/get-lost-items")
async def get_lost_items():
    if not client: return []
    try:
        response, _ = client.scroll(
            collection_name=lost_collection, limit=12, with_payload=True,
            order_by=models.OrderBy(key="upload_time", direction="desc")
        )
        return [record.payload for record in response]
    except Exception as e:
        print(f"Error getting lost items: {e}")
        return []


@app.get("/api/item-details/{collection_name}/{item_id}")
async def get_item_details(collection_name: str, item_id: str):
    if not client: raise HTTPException(status_code=503, detail="Database client is not available.")
    if collection_name not in [found_collection, lost_collection]:
        raise HTTPException(status_code=400, detail="Invalid collection name")
    try:
        point = client.retrieve(collection_name=collection_name, ids=[item_id], with_payload=True)
        if not point: raise HTTPException(status_code=404, detail="Item not found")
        return point[0].payload
    except Exception as e:
        print(f"Error getting item details: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving item details.")


# --- 8. Static File Serving ---
os.makedirs("./static", exist_ok=True)
app.mount("/images", StaticFiles(directory="uploaded_images"), name="images")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# --- 9. Run the Server ---
if __name__ == "__main__":
    # FIX: Changed reload to False so image uploads don't reboot the server
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)