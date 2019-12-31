#!/usr/bin/env python
# coding: utf-8

# 各種import等
import matplotlib.pyplot as plt
import numpy as np

# keras
import keras.datasets
import keras.utils
import keras.layers
import keras.layers.core
import keras.models

# 各種定数・変数
all_pixel = 255

# function
# データの正規化
def normaraization_func(data , bef_shape , new_shape):
    return data.reshape(bef_shape , new_shape)

# dummy 変数への変換
def conv_dummy_data(data):
    return keras.utils.np_utils.to_categorical(data)

# データ取得・加工
def get_data_func:
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

    データの正規化
    X_train = normaraization_func(X_train , 60000 , 784)  / all_pixel
    X_test  = normaraization_func(X_test , 10000 , 784 ) / all_pixel

# モデル構築
in_shape_num = 784

def create_model_func(dense_num_dict , activate_type_dict ):
    model = keras.models.Sequential()
    # Dense = 層 activation = 活性化関数
    # 隠れ層
    model.add(keras.layers.Dense(dense_num_dict[input], activation= activate_type_dict[input], input_shape=(in_shape_num,)))
    # 出力層(いくつかのカテゴライズを行う場合はsoftmaxと使う)
    model.add(keras.layers.Dense(dense_num_dict[output], activation=activate_type_dict[output]))

def  fit_and_eavl_func():

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=10)

    score = model.evaluate(X_test, y_test, verbose=1)

    return score

# 予測結果出力
def pred_result_print:

# plot
idx = 0
size = 28

color_a, color_b = np.meshgrid(range(size), range(size))
color_c = X_train[idx].reshape(size, size)
color_c = color_c[::-1,:]

plt.figure(figsize=(2.5, 2.5))
plt.xlim(0, 27)
plt.ylim(0, 27)
plt.tick_params(labelbottom="off")
plt.tick_params(labelleft="off")
plt.pcolor(color_a, color_b, color_c)
plt.gray()

# ダミーコーディング
y_train =conv_dummy_data(y_train)
y_test = conv_dummy_data(y_test)


#  arrayに1が立っている部分がその数字である、というふうに表示させている。
# 今回の場合は0,1,2,3,4,5,6,7,8,9 のうち5を選択しているので、6番目にフラグが立っている
# というようにもともと数値ではないものを数値化するのがダミーコーディング

# 1ノードにくるデータは重み付けされている。それを全て集約して、ノード内で次元圧縮する（活性化関数を使って）

# DLのモデル作成

model = keras.models.Sequential()
# Dense = 層 activation = 活性化関数
# 隠れ層
model.add(keras.layers.Dense(512, activation='sigmoid', input_shape=(784,)))
# 出力層(いくつかのカテゴライズを行う場合はsoftmaxと使う)
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=10)

score = model.evaluate(X_test, y_test, verbose=1)

score[1]

# 活性化関数の変更
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

# 活性化関数リスト
# https://keras.io/ja/activations/

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=10)
score = model.evaluate(X_test, y_test, verbose=1)
#score[0]はロス
score[1]


# 最適化関数の変更
# https://keras.io/ja/optimizers/
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=10)
score = model.evaluate(X_test, y_test, verbose=1)
score[1]

# Dropout(汎化性能up/過学習防止)
model = keras.models.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(784,)))
model.add(keras.layers.core.Dropout(0.2))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=200, verbose=1, epochs=10)
score = model.evaluate(X_test, y_test, verbose=1)
score[1]