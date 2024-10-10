# from rest_framework.decorators import api_view
# from rest_framework.response import Response
# import cv2
# from ultralytics import YOLO
# import numpy as np
# import base64

# # Carrega o modelo YOLO (substitua o caminho do seu modelo)
# model = YOLO('best.pt')

# @api_view(['POST'])
# def detect_objects(request):
#     try:
#         # Recebe a imagem enviada via POST
#         file = request.FILES['image']
#         np_img = np.frombuffer(file.read(), np.uint8)
#         img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#         # Realiza a detecção com YOLO
#         results = model(img)

#         # Processa os resultados e desenha as caixas na imagem
#         for result in results:
#             img = result.plot()

#         # Converte a imagem resultante para base64
#         _, img_encoded = cv2.imencode('.jpg', img)
#         img_base64 = base64.b64encode(img_encoded).decode('utf-8')

#         # Retorna a imagem como base64
#         return Response({"status": "success", "image": img_base64})
#     except Exception as e:
#         return Response({"status": "error", "message": str(e)})

from rest_framework.decorators import api_view
from rest_framework.response import Response
import cv2
from ultralytics import YOLO
import numpy as np
import base64

# Carrega o modelo YOLO (substitua o caminho do seu modelo)
model = YOLO('best.pt')

@api_view(['POST'])
def detect_objects(request):
    try:
        # Recebe a imagem enviada via POST
        file = request.FILES['image']
        np_img = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Realiza a detecção com YOLO
        results = model(img)

        # Processa os resultados e desenha as caixas na imagem
        for result in results:
            img = result.plot()

        # Converte a imagem resultante para base64
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        # Retorna a imagem como base64
        return Response({"status": "success", "image": img_base64})
    except Exception as e:
        return Response({"status": "error", "message": str(e)})
