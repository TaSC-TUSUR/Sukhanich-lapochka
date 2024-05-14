import matplotlib.pyplot as plt
import tensorflow.keras as krs
import numpy
from keras.datasets import cifar10
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

nw = 32
nh = 32
num_classes = 10

# Загрузка датасета CIFAR-10
(trainx, trainy), (testx, testy) = cifar10.load_data()

# Преобразование данных
all_image = (trainx / 255.0 - 0.5) * 1.999

# Задание входного слоя энкодера
encoder_input = krs.layers.Input(shape=(nw, nh, 3))

# Сверточные слои
lay = krs.layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(encoder_input)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(lay)
lay = krs.layers.Dropout(0.15)(lay)
lay = krs.layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='same')(lay)

# Преобразование в плоский вектор
lay = krs.layers.Flatten()(lay)

# Выходной кодирующий слой
lay_out_encoder = krs.layers.Dense(num_classes, activation="linear", name='den4')(lay)

# Создание сети энкодера
encoder = krs.Model(encoder_input, lay_out_encoder)

# Создание сети декодера
#decoder_input = krs.layers.Input(shape=(num_classes,))
#lay = krs.layers.Dense(256*4*4)(decoder_input)
#lay = krs.layers.Reshape(target_shape=(4, 4, 256))(lay)
#lay = krs.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(lay)
#lay = krs.layers.UpSampling2D(size=(2, 2))(lay)
#lay = krs.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(lay)
#lay = krs.layers.UpSampling2D(size=(2, 2))(lay)
#lay_out_decoder = krs.layers.Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')(lay)

# Создание сети декодера
decoder_input = krs.layers.Input(shape=(num_classes,))
lay = krs.layers.Dense(256*8*8)(decoder_input)  # Изменено для увеличения размера
lay = krs.layers.Reshape(target_shape=(8, 8, 256))(lay)  # Изменено для увеличения размера
lay = krs.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(lay)
lay = krs.layers.UpSampling2D(size=(2, 2))(lay)
lay = krs.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(lay)
lay = krs.layers.UpSampling2D(size=(2, 2))(lay)
lay_out_decoder = krs.layers.Conv2DTranspose(3, (3, 3), activation='tanh', padding='same')(lay)

# Создание сети декодера
decoder = krs.Model(decoder_input, lay_out_decoder)

# Объединение обеих сетей в автоэнкодер
lay_out = decoder(lay_out_encoder)
autoencoder = krs.Model(encoder_input, lay_out)

# Сохранение модели в файле в виде изображения
#krs.utils.plot_model(autoencoder, to_file='./out/autoencoder.png', show_shapes=True)

# Компиляция модели автоэнкодера
autoencoder.compile(loss='mean_squared_error', optimizer=krs.optimizers.Adam(learning_rate=0.0002),
                    metrics=['accuracy', krs.metrics.Precision(), krs.metrics.Recall()])

# Обучение модели
ep = 20
autoencoder.fit(x=all_image, y=all_image, batch_size=4000, epochs=ep)

# Сохранение модели энкодера
encoder.save('encoder_model.keras')

# Получение выхода автоэнкодера
index = numpy.random.randint(0, len(all_image), 9)
#out_img = autoencoder.predict(all_image[index])

# Вывод изображений на графике
#fig = plt.figure(figsize=(5, 5))
#for i in range(3):
#    for j in range(3):
#        ax = fig.add_subplot(3, 3, i*3+j+1)
#        ax.imshow(out_img[i*3+j])
#plt.show()

out_img = autoencoder.predict(all_image[index])

# Нормализация изображений к диапазону [0, 1]
out_img = (out_img + 1) / 2

# Вывод изображений на графике
fig = plt.figure(figsize=(5, 5))
for i in range(3):
    for j in range(3):
        ax = fig.add_subplot(3, 3, i * 3 + j + 1)
        ax.imshow(out_img[i * 3 + j])
plt.show()

# Реализация работы с энкодером для получения скрытого кодового слоя
out_vec = encoder.predict(all_image)

from scipy.cluster.vq import kmeans2

# Получение скрытого кодового слоя с помощью энкодера для всех изображений
out_vec = encoder.predict(all_image)
out_vec_save = out_vec

# Получение центроидов кластеров для 10 кластеров
centroid, label = kmeans2(out_vec, 10, minit='++')

# Получение центроидов кластеров для 2 кластеров
centroid1, label1 = kmeans2(out_vec, 2, minit='++')

# Вычисление координат кластера как разность с центроидом
out_vec1 = (out_vec - centroid1[0]) ** 2
out_vec2 = (out_vec - centroid1[1]) ** 2

# Вычисление среднего значения
outm = out_vec1.mean(axis=1)
outstd = out_vec2.mean(axis=1)

coutm = centroid.mean(axis=1)
coutstd = centroid.mean(axis=1)

# Отрисовка кластеров
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
for i in range(num_classes):
    mask = label == i
    ax.scatter(out_vec[mask, 0], out_vec[mask, 1])
    plt.text(centroid[i, 0], centroid[i, 1], i, fontdict=None)
plt.show()

# Вычисление метрик кластеризации
silhouette = silhouette_score(out_vec, label)
davies_bouldin = davies_bouldin_score(out_vec, label)
calinski_harabasz = calinski_harabasz_score(out_vec, label)

# Вывод метрик кластеризации
print("Cluster Validity Metrics:")
print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", davies_bouldin)
print("Calinski-Harabasz Index:", calinski_harabasz)

# Подсчет количества точек данных в каждом кластере
cluster_sizes = numpy.bincount(label)

# Построение гистограммы размеров кластеров
plt.figure(figsize=(8, 6))
plt.bar(range(len(cluster_sizes)), cluster_sizes, color='skyblue')
plt.xlabel('Cluster')
plt.ylabel('Number of Data Points')
plt.title('Histogram of Cluster Sizes')
plt.show()
