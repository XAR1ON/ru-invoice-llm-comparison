import json
import os
from pathlib import Path
import pytesseract
import cv2

# --- НАСТРОЙКИ ---
CURRENT_DIR = Path(__file__).resolve().parent 
BASE_DIR = CURRENT_DIR.parent
INPUT_JSONL = BASE_DIR / "data" / "dataset.jsonl"         # Ваш исходный файл
OUTPUT_JSONL = BASE_DIR / "data" / "dataset_final.jsonl"   # Файл, куда запишем всё вместе
IMAGES_DIR = BASE_DIR / "data" / "raw_img"                   # Папка с фото
OCR_DIR = BASE_DIR / "ocr_output"                  # Папка для сырых текстов OCR

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.makedirs(OCR_DIR, exist_ok=True)

def preprocess_image(image_path):
    """Улучшаем картинку для OCR"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh

def json_to_ideal_text(gt):
    """Генерирует идеальный текст из Ground Truth (JSON)"""
    text_parts = []
    text_parts.append(f"Счет-фактура № {gt['document_info']['number']}")
    
    seller = gt['parties']['seller']
    text_parts.append(f"Продавец: {seller['name']}")
    text_parts.append(f"ИНН/КПП продавца: {seller['inn']}/{seller['kpp']}")
    text_parts.append(f"Адрес продавца: {seller['address']}")
    
    buyer = gt['parties']['buyer']
    text_parts.append(f"Покупатель: {buyer['name']}")
    text_parts.append(f"Адрес покупателя: {buyer['address']}")
    
    text_parts.append("\nТовары (описание, цена, кол-во, сумма):")
    for item in gt['terms']['items']:
        text_parts.append(f"{item.get('item_name', '')} | {item.get('price', '')} | {item.get('quantity', '')} | {item.get('total_amount', '')}")
        
    return "\n".join(text_parts)

def process_dataset():
    final_records = []
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        record = json.loads(line)
        
        # 1. Определяем имя файла (картинки и текста)
        # Если в JSONL есть text_path, берем имя оттуда. Иначе ищем по-другому.
        base_name = "unknown"
        if "text_path" in record:
            base_name = Path(record["text_path"]).stem
        elif "image_path" in record:
            base_name = Path(record["image_path"]).stem
        
        img_path = os.path.join(IMAGES_DIR, f"{base_name}.jpg") # или .png
        ocr_path = os.path.join(OCR_DIR, f"{base_name}.txt")
        
        # 2. ДЕЛАЕМ OCR (если файла нет)
        if os.path.exists(ocr_path):
            print(f"[OCR] {base_name}.txt уже существует. Читаем...")
            with open(ocr_path, 'r', encoding='utf-8') as txt_f:
                ocr_text = txt_f.read()
        else:
            print(f"[OCR] Распознаем {base_name}...")
            try:
                processed_img = preprocess_image(img_path)
                ocr_text = pytesseract.image_to_string(processed_img, lang='rus+eng', config='--psm 4')
                with open(ocr_path, 'w', encoding='utf-8') as txt_f:
                    txt_f.write(ocr_text)
            except Exception as e:
                print(f"  ОШИБКА OCR: {e}")
                ocr_text = ""
                
        # 3. ГЕНЕРИРУЕМ ИДЕАЛЬНЫЙ ТЕКСТ
        ideal_text = json_to_ideal_text(record["ground_truth"])
        
        # 4. СОБИРАЕМ НОВУЮ ЗАПИСЬ
        final_record = {
            "id": base_name,
            "prompt": record["prompt"],
            "ground_truth": record["ground_truth"],
            "ocr_text": ocr_text,
            "ideal_text": ideal_text
        }
        final_records.append(final_record)
        
    # Сохраняем в новый файл
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for rec in final_records:
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"\nГотово! Все данные (OCR + идеальный текст + эталон) собраны в {OUTPUT_JSONL}")

if __name__ == "__main__":
    process_dataset()