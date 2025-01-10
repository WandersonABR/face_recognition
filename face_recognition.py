import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model
from sklearn.preprocessing import Normalizer
import cv2

# Categorizando as pessoas
pessoa = ["Ana C", "Desconhecidos", "Wanderson"]
num_classes = len(pessoa)

# Inicia a captura de vídeo, 0 é a primeira camera conectada
cap = cv2.VideoCapture(0)

# Detector de face
detector = MTCNN()
# Modelo para transformar a face em embeddings
facenet = load_model("facenet_keras.h5")
# Meu modelo treinado para reconhecer eu, minha esposa e desconhecidos
model = load_model("faces_desc.h5")

# Função para extrair a face do vídeo
def extract_face(image, box, required_size=(160, 160)):
    pixels = np.asarray(image)
    x1, y1, width, height = box
    x2, y2 = x1 + width, y1 + height

    # para ter certeza que as coordenadas sejam válidas
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(pixels.shape[1], x2), min(pixels.shape[0], y2)

    face = pixels[y1:y2, x1:x2]
    # Transforma a face em array
    image = Image.fromarray(face)
    # Redimensiona a face
    image = image.resize(required_size)
    # Retorna a imagem como array
    return np.asarray(image)

# Função para transformar o rosto em embedding, assinatura numerica
def get_embedding(facenet, face_pixels):
    # Convertendo para ponto flutuante
    face_pixels = face_pixels.astype('float32')
    # Normalizando para garantir que os pixels tem valores entre -1 e 1
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # Gerando os embeddings 128 dimensões
    samples = np.expand_dims(face_pixels, axis=0)

    # Fazendo a predição da assinatura facial
    yhat = facenet.predict(samples)
    # Retornando o vetor gerado
    return yhat[0]

# Loop principal
while True:
    # Verificando a captura da camera
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame.")
        break
    # Detectando a face com a biblioteca MTCNN
    faces = detector.detect_faces(frame)
    for face in faces:
        # Verifica a confiança do detector
        confidence = face["confidence"] * 100
        if confidence >= 98:
            x1, y1, w, h = face["box"]
            try:
                # Para recortar e redimensionar o rosto
                face_pixels = extract_face(frame, face["box"])
                # Gera o embedding do rosto
                emb = get_embedding(facenet, face_pixels)
                sample = np.expand_dims(emb, axis=0)

                # Normalizando os embeddings
                norm = Normalizer(norm="l2")
                sample = norm.transform(sample)

                # Previsão do tipo de classe
                probas = model.predict(sample)
                # Pega a classe com maior probabilidade
                classe = np.argmax(probas, axis=-1)[0]
                # Confiança da predição
                prob = probas[0][classe] * 100
                # Deixei a acuracia alta por causa que eu preciso de precisão
                if prob >= 98:
                    # Para definir a cor para conhecidos e desconhecidos
                    color = (192, 255, 119) if classe != 1 else (224, 43, 100)
                    user = pessoa[classe].upper()
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
                    cv2.putText(frame, user, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except Exception as e:
                print(f"Erro ao processar a face: {e}")

    # Exibe o frame
    cv2.imshow("FACE RECOGNITION", frame)

    # ESC para sair
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
