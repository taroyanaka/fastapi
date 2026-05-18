import io
import cv2
import numpy as np
import easyocr
import threading
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import json

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORSを許可する設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 自分のサイトだけに絞る場合は ["https://your-site.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"Hello": "World"}

# OCRリーダーの初期化（起動時に一度だけ）
reader = easyocr.Reader(['ja', 'en'], gpu=False)

# 同時実行を防ぐためのロックオブジェクト
processing_lock = threading.Lock()

def process_masking(image_bytes, target_list):
    """
    重い画像処理ロジック（中身は前回のまま）
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return None

    # EasyOCRの実行（ここが一番メモリを食う）
    results = reader.readtext(image)

    for (bbox, text, prob) in results:
        cleaned_text = text.replace(" ", "").replace("　", "")
        for target in target_list:
            if target in cleaned_text:
                start_idx = cleaned_text.find(target)
                while start_idx != -1:
                    tl, tr, br, bl = bbox
                    raw_len = len(text)
                    full_width = tr[0] - tl[0]
                    char_width = full_width / raw_len
                    
                    word_start_x = int(tl[0] + (char_width * start_idx))
                    word_end_x = int(tl[0] + (char_width * (start_idx + len(target))))
                    
                    padding = 4
                    start_x = max(int(tl[0]), word_start_x - padding)
                    end_x = min(int(tr[0]), word_end_x + padding)
                    
                    height = br[1] - tl[1]
                    start_y = max(0, int(tl[1] - height * 0.1))
                    end_y = min(image.shape[0], int(br[1] + height * 0.1))

                    cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 0), -1)
                    start_idx = cleaned_text.find(target, start_idx + 1)

    res, im_png = cv2.imencode(".png", image)
    return im_png.tobytes()

@app.post("/mask-image")
def mask_image(
    file: UploadFile = File(...), 
    target_words: str = Form(...)
):
    """
    asyncを付けないことで、FastAPIはリクエストを順番に処理します。
    さらにLockを使い、OCR処理が重なっている間のメモリ爆発を徹底的に防ぎます。
    """
    # 1. パラメータのパース
    target_list = json.loads(target_words)
    
    # 2. 画像の読み込み（ここはメモリに載る）
    image_bytes = file.file.read()
    
    # 3. ロックを取得して1枚ずつ処理
    # 他のリクエストが処理中の場合、ここで待機（キューイング）されます
    with processing_lock:
        output_bytes = process_masking(image_bytes, target_list)
    
    if output_bytes is None:
        return {"error": "画像の処理に失敗しました。"}

    return StreamingResponse(io.BytesIO(output_bytes), media_type="image/png")