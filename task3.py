from icrawler.builtin import GoogleImageCrawler


# import os module
import os
from PIL import Image 
from imutils import paths

google_crawler = GoogleImageCrawler(
    feeder_threads=4,
    parser_threads=4,
    downloader_threads=8, 
    storage={'root_dir':'D:\\парсер\\celebi,'})

#print('Количество картинок?')
#count = 1000
#print('Запрос')
#name = str(input())

#google_crawler.crawl(keyword=name,max_num= count)

# directory path
#path_name = r'D:\\парсер\\celebi,'
 
# Sort list of files based on last modification time in ascending order using list comprehension
#name_list = os.listdir(path_name)
#full_list = [os.path.join(path_name,i) for i in name_list]
#sorted_list = sorted(full_list, key=lambda x: os.path.getmtime(x))
 
#count = 553
# loop through files and rename
#for file in sorted_list:
    # print(file)
    #prefix = "000"
   # counter = str(count).zfill(2)
   # new = prefix + counter + '.jpg'  # new file name
   # src = os.path.join(path_name, file)  # file source
   # dst = os.path.join(path_name, new)  # file destination
   # os.rename(src, dst)
   # count +=1
# loop through files and rename
#for file in paths.list_images(path_name):

  # Opening image using PIL Image 
 # im1 = Image.open(file).convert('RGB')
 # im1 =  im1.rotate(90)
  # Saving the image with quality=95% in jpg format 
#  im1.save(file)


# импортируем бэкенд Agg из matplotlib для сохранения графиков на диск
import matplotlib
matplotlib.use("Agg")

# Подключаем необходимые модули и пакеты
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import cv2
import os

# инициализируем данные и метки
print("[INFO] loading images...")
data = []
labels = []

# Собираем список путей к каждому изображению и перемешиваем их
imagePaths = sorted(list(paths.list_images("./Dataset")))
random.shuffle(imagePaths)
# цикл по изображениям
for imagePath in imagePaths:
	
	image = cv2.imread(imagePath) # загружаем изображение
	image = cv2.resize(image, (32, 32)).flatten() # меняем его разрешение на 32x32 пикселей (без учета соотношения сторон),
																								# сглаживаем его в 32x32x3=3072 пикселей
	data.append(image) # добавляем в список

	label = imagePath.split(os.path.sep)[-2] 	# извлекаем метку класса из пути к изображению (метка класса зависит от имени папки)
	labels.append(label) # обновляем список меток

# масштабируем интенсивности пикселей в диапазон [0, 1] (Нормализация данных)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# разбиваем данные на обучающую и тестовую выборки, используя 75%
# данных для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# конвертируем метки из целых чисел в векторы (для 2х классов при
# бинарной классификации вам следует использовать функцию Keras
# "to_categorical" вместо "LabelBinarizer" из scikit-learn, которая не возвращает вектор)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# определим архитектуру 3072-1024-512-3 с помощью Keras
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="relu"))
model.add(Dense(512, activation="relu"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# инициализируем скорость обучения и общее число эпох
INIT_LR = 0.01

print('Количество эпох')
EPOCHS = int(input())

# компилируем модель, используя SGD как оптимизатор и категориальную
# кросс-энтропию в качестве функции потерь (для бинарной классификации
# следует использовать binary_crossentropy)
print("[INFO] training network...")
opt = SGD(learning_rate =INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])

H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=750)

# оцениваем нейросеть
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# строим графики потерь и точности
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["accuracy"], label="train_acc")
plt.plot(N, H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('plot.png')

# сохраняем модель и метки классов в бинарном представлении на диск
print("[INFO] serializing network and label binarizer...")
model.save("model.h5", save_format="h5")
f = open("label_bin", "wb")
f.write(pickle.dumps(lb))
f.close()

