from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
from .document_manager import insert, delete
from .search_engine import search
import shutil

app = FastAPI()

UPLOAD_DIR = 'search_module/tmp_uploads/'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post('/insert')
def insert_file(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.pdf', '.txt']:
        raise HTTPException(status_code=400, detail='Only PDF and TXT supported')
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)
    try:
        file_id = insert(temp_path)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    return {'status': 'ok', 'file_id': file_id}

@app.post('/search')
def search_query(query: str = Form(...)):
    try:
        results = search(query)
    except Exception as e:
        import traceback
        print("SEARCH ERROR:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})
    return {'results': results}

@app.post('/delete')
def delete_file(file_id: str = Form(...)):
    try:
        delete(file_id)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    return {'status': 'deleted', 'file_id': file_id} 