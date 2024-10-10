# import requests
# import base64
# import matplotlib.pyplot as plt
# import io
# from PIL import Image
# import cv2

# # URL da API Django
# url = 'http://127.0.0.1:8000/api/detect/'

# # Inicializa a captura de vídeo da webcam (0 é geralmente o ID da webcam padrão)
# cap = cv2.VideoCapture(0)

# # Verifique se a câmera está aberta corretamente
# if not cap.isOpened():
#     print("Erro ao abrir a câmera")
#     exit()

# # Loop para capturar e enviar frames da câmera
# while True:
#     # Captura o frame da câmera
#     ret, frame = cap.read()

#     if not ret:
#         print("Falha ao capturar o frame")
#         break

#     # Converte o frame para JPEG
#     _, img_encoded = cv2.imencode('.jpg', frame)

#     # Prepara o arquivo para envio
#     file = {'image': img_encoded.tobytes()}

#     # Envia o frame para a API
#     response = requests.post(url, files=file)

#     # Verifica se a resposta foi bem-sucedida
#     if response.status_code == 200:
#         response_data = response.json()
#         print("Status:", response_data['status'])

#         if response_data['status'] == 'success':
#             # Decodificar a imagem em base64
#             img_data = base64.b64decode(response_data['image'])

#             # Converter bytes para uma imagem usando PIL
#             image = Image.open(io.BytesIO(img_data))

#             # Exibir a imagem com matplotlib
#             plt.imshow(image)
#             plt.axis('off')  # Ocultar os eixos
#             plt.show()

#         else:
#             print("Erro na resposta da API:", response_data['message'])
#     else:
#         print("Falha na resposta da API:", response.status_code)

#     # Se a tecla 'q' for pressionada, o loop é encerrado
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Libera a câmera e fecha janelas
# cap.release()


from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

# Carrega o modelo YOLO (substitua o caminho do seu modelo)
model = YOLO('best.pt')

# Inicializa a captura de vídeo da webcam (0 é geralmente o ID da webcam padrão)
cap = cv2.VideoCapture(0)

track_history = defaultdict(lambda: [])
seguir = True
deixar_rastro = True

# Verifica se a câmera está aberta corretamente
if not cap.isOpened():
    print("Erro ao abrir a câmera")
    exit()

# Inicializa a exibição do matplotlib
fig, ax = plt.subplots()
img_plot = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))  # Inicia com uma imagem vazia
plt.axis('off')  # Desativa os eixos para otimizar a renderização

while True:
    # Captura o frame da câmera
    ret, img = cap.read()

    if not ret:
        print("Falha ao capturar o frame")
        break

    if seguir:
        results = model.track(img, persist=True)
    else:
        results = model(img)

    # Processa os resultados e desenha as caixas na imagem
    for result in results:
        img = result.plot()

        if seguir and deixar_rastro:
            try:
                # Obtém as caixas e os track IDs
                boxes = result.boxes.xywh.cpu()
                track_ids = result.boxes.id.int().cpu().tolist()

                # Desenha as linhas de rastreamento
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # ponto central (x, y)
                    if len(track) > 30:  # mantem 30 frames de rastro
                        track.pop(0)

                    # Desenha as linhas de rastreamento
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)
            except:
                pass

    # Converte de BGR para RGB para exibir com matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Atualiza o frame no matplotlib sem recriar a figura
    img_plot.set_data(img_rgb)
    fig.canvas.draw_idle()  # Desenha sem bloquear o fluxo
    plt.pause(0.001)  # Pequena pausa para manter a fluidez

    # Encerra o loop se a tecla 'q' for pressionada
    try:
        if plt.waitforbuttonpress(0.001):  # aguarda 0,001 segundos por entrada
            break
    except KeyboardInterrupt:
        break

# Libera a câmera e fecha as janelas
cap.release()
plt.close()


# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt
# from collections import defaultdict
# import numpy as np

# # Carrega o modelo YOLO (substitua o caminho do seu modelo)
# model = YOLO('best.pt')

# # Inicializa a captura de vídeo da webcam (0 é geralmente o ID da webcam padrão)
# cap = cv2.VideoCapture(0)

# track_history = defaultdict(lambda: [])
# seguir = True
# deixar_rastro = True

# # Verifica se a câmera está aberta corretamente
# if not cap.isOpened():
#     print("Erro ao abrir a câmera")
#     exit()

# # Configura o matplotlib para manter a janela aberta
# plt.ion()  # Ativa o modo interativo
# fig, ax = plt.subplots()

# # IDs de interesse: Hardhat e No-Hardhat
# ids_interesse = [2, 3]

# while True:
#     # Captura o frame da câmera
#     ret, img = cap.read()

#     if not ret:
#         print("Falha ao capturar o frame")
#         break

#     if seguir:
#         results = model.track(img, persist=True)
#     else:
#         results = model(img)

#     # Processa os resultados e desenha as caixas na imagem
#     for result in results:
#         # Itera pelas caixas detectadas
#         for box in result.boxes:
#             track_id = int(box.id.item())  # Obtém o ID de rastreamento

#             # Filtra apenas os objetos com track_id 2 e 3
#             if track_id in ids_interesse:
#                 cls_id = int(box.cls)  # ID da classe detectada
#                 label = result.names[cls_id]  # Nome da classe

#                 # Mostra no console os IDs e as classes
#                 print(f"ID {track_id} - Classe: {label}")

#                 # Obtém as coordenadas da caixa delimitadora
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

#                 # Desenha a caixa ao redor do objeto detectado
#                 cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv2.putText(img, f"ID {track_id} - {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

#                 # Rastreia o histórico do objeto
#                 track = track_history[track_id]
#                 track.append((float((x1 + x2) / 2), float((y1 + y2) / 2)))  # ponto central (x, y)
#                 if len(track) > 30:  # mantém 30 frames de rastro
#                     track.pop(0)

#                 # Desenha as linhas de rastreamento
#                 points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
#                 cv2.polylines(img, [points], isClosed=False, color=(230, 0, 0), thickness=5)

#     # Converte de BGR para RGB para exibir com matplotlib
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#     # Exibe o frame usando matplotlib
#     ax.imshow(img_rgb)
#     plt.draw()
#     plt.pause(0.01)  # Pequena pausa para manter a fluidez

#     # Encerra o loop se a tecla 'q' for pressionada
#     try:
#         if plt.waitforbuttonpress(0.01):  # aguarda 0,01 segundos por entrada
#             break
#     except KeyboardInterrupt:
#         break

# # Libera a câmera e fecha as janelas
# cap.release()
# plt.ioff()  # Desativa o modo interativo do matplotlib
# plt.close()
