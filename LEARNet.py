import tensorflow as tf
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten,BatchNormalization
from tensorflow.keras.activations import relu,softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix  # 生成混淆矩阵函数
import matplotlib.pyplot as plt  # 绘图库
import seaborn as sns
from LoadData import load_test_data,load_valid_data,load_train_data
logdir = './log'

class self_LERANet(Model):
    def __init__(self):
        super(self_LERANet,self).__init__()
        self.conv1 = Conv2D(filters=16,kernel_size=(3,3),strides=2,activation=relu)
        self.conv2_1 = Conv2D(filters=16,kernel_size=(1,1),strides=2,activation=relu)
        self.conv2_2 = Conv2D(filters=16,kernel_size=(1,1),strides=2,activation=relu)
        self.conv2_3 = Conv2D(filters=16, kernel_size=(1, 1), strides=2, activation=relu)
        self.conv2_4 = Conv2D(filters=16, kernel_size=(1, 1), strides=2, activation=relu)
        self.conv3_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation=relu)
        self.conv3_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation=relu)
        self.conv3_3 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation=relu)
        self.conv3_4 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, activation=relu)
        self.conv4_1 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation=relu)
        self.conv4_2 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation=relu)
        self.conv4_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation=relu)
        self.conv4_4 = Conv2D(filters=64, kernel_size=(5, 5), strides=2, activation=relu)
        self.conv5 = Conv2D(filters=256,kernel_size=(3,3), strides=2, activation=relu)
        self.fallten = Flatten()
        self.fc1 = Dense(units=256,activation=relu)
        self.fc2 = Dense(units=7, activation=softmax)

    def call(self,input):
        out1 = self.conv1(input)
#         print("out1.shape:",out1.shape)
        out2_1 = self.conv2_1(out1)
#         print("out2_1.shape:", out2_1.shape)
        out2_2 = self.conv2_2(out1)
#         print("out2_2.shape:", out2_1.shape)
        out2_3 = self.conv2_3(out1)
#         print("out2_3.shape:", out2_1.shape)
        out2_4 = self.conv2_4(out1)
#         print("out2_4.shape:", out2_1.shape)
        add2_2 = out2_1 + out2_2
#         print("add2_2.shape:", add2_2.shape)
        add2_3 = out2_3 + out2_4
#         print("add2_3.shape:", add2_3.shape)
        out3_1 = self.conv3_1(out2_1)
#         print("out3_1.shape:", out3_1.shape)
        out3_2 = self.conv3_2(add2_2)
#         print("out3_2.shape:", out3_2.shape)
        out3_3 = self.conv3_3(add2_3)
#         print("out3_3.shape:", out3_3.shape)
        out3_4 = self.conv3_4(out2_4)
#         print("out3_4.shape:", out3_4.shape)
        add3_2 = out3_1 + out3_2
#         print("add3_2.shape:", add3_2.shape)
        add3_3 = out3_3 + out3_4
#         print("add3_2.shape:", add3_3.shape)
        out4_1 = self.conv4_1(out3_1)
#         print("out4_1.shape:", out4_1.shape)
        out4_2 = self.conv4_2(add3_2)
#         print("out4_2.shape:", out4_2.shape)
        out4_3 = self.conv4_3(add3_3)
#         print("out4_3.shape:", out4_3.shape)
        out4_4 = self.conv4_4(out3_4)
#         print("out4_4.shape:", out4_4.shape)
        concat = tf.concat([out4_1,out4_2,out4_3,out4_4],axis=3)
#         print("concat.shape:", concat.shape)
        concat_lrn = tf.nn.lrn(concat)
#         print("concat_lrn.shape:", concat_lrn.shape)
        out5 = self.conv5(concat_lrn)
#         print("out5.shape:", out5.shape)
        out6 = self.fallten(out5)
        out6 = self.fc1(out6)
#         print("out6.shape:", out6.shape)
        out7 = self.fc2(out6)
#         print("out7.shape:", out7.shape)
        return out7

# 绘制混淆矩阵
def plot_cm(labels, predictions, epoch=1):
    cm = confusion_matrix(labels, predictions)
    print(cm)
    plt.figure(figsize=(7, 7))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(epoch))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('./cm/{}@.jpg'.format(epoch))

def main():
    LERANet = self_LERANet()
    LERANet.build(input_shape=(None, 112, 112, 3))
    optimizer = Adam(lr=1e-4)
    variables = LERANet.trainable_variables
    # 获取数据集
    train_data,train_label = load_train_data()
    valid_data, valid_label = load_valid_data()
    test_data, test_label = load_test_data()

    # 切割数据集并且打乱
    train_datasets = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    valid_datasets = tf.data.Dataset.from_tensor_slices((valid_data, valid_label))
    test_datasets = tf.data.Dataset.from_tensor_slices((test_data, test_label))
    # 打乱,并分批
    train_datasets = train_datasets.shuffle(5000)
    valid_datasets = valid_datasets.shuffle(5000)
    test_datasets = test_datasets.shuffle(20000)
    
    valid_datasets = valid_datasets.batch(32)
    train_datasets = train_datasets.batch(32)
    test_datasets = test_datasets.batch(32)

    for epoch in range(100):
        ground_list = []
        predic_list = []
        loss_list = []
        for step, (x, y) in enumerate(train_datasets):
            with tf.GradientTape() as tape:
                # [b, 112, 112, 3] => [b, 7]
                prob = LERANet(x)
                # [b] => [b, 7]
                y = tf.cast(y,dtype=tf.int32)
                y_onehot = tf.one_hot(y, depth=7)
                # 计算误差函数
                loss = tf.losses.categorical_crossentropy(y_onehot, prob, from_logits=False)
                loss = tf.reduce_mean(loss)
                loss_list.append(loss)

            grads = tape.gradient(loss, variables)
            optimizer.apply_gradients(zip(grads, variables))
            print(epoch, " epoches ", step, " steps ", " loss: ", float(loss))

        # 可视化
        summury_writer = tf.summary.create_file_writer(logdir)
        with summury_writer.as_default():
            tf.summary.scalar('Loss', float(min(loss_list)), step=epoch)


        # 做测试
        total_num = 0
        total_correct = 0
        for x, y in valid_datasets:
            prob = LERANet(x)
            pred = tf.argmax(prob, axis=1)

            pred = tf.cast(pred, dtype=tf.int32)    # 还記得吗pred类型为int64,需要转换一下。
            y = tf.cast(y,dtype=tf.int32)

            predic_list.append(pred)
            ground_list.append(y)

            # 拿到预测值pred和真实值比较。
            correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
            correct = tf.reduce_sum(correct)

            total_num += x.shape[0]
            total_correct += int(correct)

        acc = total_correct / total_num
        print(epoch, 'acc:', acc)

        predic = tf.concat([predic_list[i] for i in range(len(predic_list))], axis=0)
        ground = tf.concat([ground_list[i] for i in range(len(ground_list))], axis=0)
        predic = list(predic.numpy())
        ground = list(ground.numpy())
        print("predic:", predic)
        print("ground:", ground)

        plot_cm(ground, predic, epoch)

        with summury_writer.as_default():
            tf.summary.scalar('TOP-1 Accuracy', float(acc), step=epoch)


    test_ground_list = []
    test_predic_list = []
    for x, y in test_datasets:
        prob = LERANet(x)
        pred = tf.argmax(prob, axis=1)

        pred = tf.cast(pred, dtype=tf.int32)  # 还記得吗pred类型为int64,需要转换一下。
        y = tf.cast(y, dtype=tf.int32)
        
        test_predic_list.append(pred)
        test_ground_list.append(y)

        # 拿到预测值pred和真实值比较。
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    print(epoch, 'acc:', acc)
    predic = tf.concat([test_predic_list[i] for i in range(len(test_predic_list))], axis=0)
    ground = tf.concat([test_ground_list[i] for i in range(len(test_ground_list))], axis=0)

    plot_cm(ground, predic, epoch=2020)

    LERANet.save('LEARNet.h5')

if __name__ == '__main__':
    main()