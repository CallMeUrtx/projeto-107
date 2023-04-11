import cv2

# Carrega a imagem
img = cv2.imread('imagem.jpg')

# Converte a imagem para escala de cinza
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Carrega o classificador pré-treinado para detecção de objetos
object_cascade = cv2.CascadeClassifier('cascade.xml')

# Detecta objetos na imagem
objects = object_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# Desenha retângulos em torno dos objetos detectados
for (x,y,w,h) in objects:
    cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

# Mostra a imagem com os objetos detectados
cv2.imshow('Detecção de Objetos', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
