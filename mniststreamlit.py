#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Desc  :     
@Author:   zhangchang
@Date  :   2021/11/20 15:58
'''

import ndraw
import streamlit as st
import tensorflow as tf
import streamlit.components.v1 as components


# 设置随机种子
tf.random.set_seed(1)


# 查看gpu是否有效
# print(tf.test.is_gpu_available())
# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# tf.config.experimental.set_memory_growth(gpu[0], True)

class TrainCallback(tf.keras.callbacks.Callback):
    def __init__(self, test_x, test_y):
        super(TrainCallback, self).__init__()
        self.test_x = test_x
        self.test_y = test_y

    def on_train_begin(self, logs=None):
        st.header("训练汇总")
        self.summary_line = st.area_chart()

        st.subheader("训练进度")
        self.process_text = st.text("0/{}".format(self.params['epochs']))
        self.process_bar = st.progress(0)

        st.subheader('loss曲线')
        self.loss_line = st.line_chart()

        st.subheader('accuracy曲线')
        self.acc_line = st.line_chart()

    def on_epoch_end(self, epoch, logs=None):
        self.loss_line.add_rows({'train_loss': [logs['loss']], 'val_loss': [logs['val_loss']]})
        self.acc_line.add_rows({'train_acc': [logs['accuracy']], 'val_accuracy': [logs['val_accuracy']]})
        self.process_bar.progress(epoch / self.params['epochs'])
        self.process_text.empty()
        self.process_text.text("{}/{}".format(epoch, self.params['epochs']))

    def on_batch_end(self, epoch, logs=None):
        if epoch % 10 == 0 or epoch == self.params['epochs']:
            self.summary_line.add_rows({'loss': [logs['loss']], 'accuracy': [logs['accuracy']]})


@st.cache(allow_output_mutation=True)
def get_data(is_onehot = False):
    # 根据自己的训练数据进行加载
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train/255.0
    x_test = x_test/255.0
    if is_onehot:
        y_train = tf.one_hot(y_train,10)
        y_test = tf.one_hot(y_test,10)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model


if __name__ == '__main__':
    st.title("训练xx模型")
    if st.sidebar.button('开始'):
        (x_train, y_train), (x_test, y_test) = get_data(is_onehot=True)

        st.text("train size: {} {}".format(x_train.shape, y_train.shape))
        st.text("test size: {} {}".format(x_test.shape, y_test.shape))

        model = build_model()
        with st.expander("查看模型"):
            components.html(ndraw.render(model,init_x=200, flow=ndraw.VERTICAL), height=1000, scrolling=True)
        model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.001),metrics=["accuracy"])
        model.fit(x_train, y_train, batch_size=128, validation_data=(x_test, y_test), epochs=10, verbose=1,callbacks=[TrainCallback(x_test, y_test)])
        st.success('训练结束')

    if st.sidebar.button('停止'):
        st.stop()
