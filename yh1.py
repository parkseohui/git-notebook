#import tensorflow되는지 테스트
import tensorflow as tf

a = tf.constant('1')

import numpy as np

num_points = 200#200개의 점을 생성함

vectors_set = []# [1,2,3,4,5]
                # ['t','c','ddfd']
                # [[1,2,3],
                #  [5,6,7]]

for i in range(num_points):
    x = np.random.normal(5,5) + 15
    y = x*1000 + (np.random.normal(0,3))*1000
    vectors_set.append([x,y])

vectors_set#(200 X 2)

x_data = [v[0] for v in vectors_set]#리스트 comprehension
y_data = [v[1] for v in vectors_set]

x_data = []
y_data = []
for v in vectors_set:
    x_data.append(v[0])
    y_data.append(v[1])

y_data
x_data

import matplotlib.pyplot as plt

plt.plot(x_data,y_data,'ro')

#1차원 값, -1 ~ 1 로 랜덤하게 초기화
random_val = tf.random_uniform([1],-100.0,-5.0)

#w에 값 넣음.
w = tf.Variable(random_val)

#1차원 0으로 초기화
b = tf.Variable(tf.zeros([1]))

# y = ax + b
# y_pred(y) = w(a) x(x_data) + b(b)
y_pred = w * x_data + b

#전체 평균 ( (예측값(y_pred) - 실제값(y_data))^2 )
loss = tf.reduce_mean(tf.square(y_pred - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.002)

train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()

sess.run(init)

for step in range(100):
    sess.run(train)
    print(step, sess.run(w),sess.run(b))
    print(step, sess.run(loss))
    plt.plot(x_data,y_data,'ro')
    plt.plot(x_data,sess.run(w)*x_data + sess.run(b))
    plt.show()

