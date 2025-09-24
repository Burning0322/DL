import tensorflow as tf

with tf.device("/GPU:0"):
        
    mnist = tf.keras.datasets.mnist

    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train,x_test = x_train/255.0, x_test/255.0

    #x_train 和 x_test 要除于255.0 归一化到0-1之间.而y_train和y_test不需要归一化因为他们是label

    # imgs = x_test[:3]
    # labels = y_test[:3]
    # print("labels:",labels)

    # import matplotlib.pyplot as plt

    # plt.imshow(imgs[0], cmap="gray")
    # plt.show()

    #这里的...表示的是比如x_train = x_train[:,:,:,tf.newaxis] 这样的意思
    x_train = x_train[...,tf.newaxis]
    x_test = x_test[...,tf.newaxis]

    train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).shuffle(10000).batch(32)
    test = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)
    print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

class Model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv1d = tf.keras.layers.Conv2D(6,3,activation='relu')
        self.maxpool = tf.keras.layers.MaxPool2D()
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(120,activation='relu')
        self.dense2 = tf.keras.layers.Dense(84,activation='relu')
        self.dense3 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1d(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

model = Model()

#这个不是one-hot编码的
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
#这个是one-hot编码的
#loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

#pytorch里的是手动计算损失的loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()

    for images, labels in train:
        train_step(images, labels)

    for test_images, test_labels in test:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(),
                          train_accuracy.result()*100,
                          test_loss.result(),
                          test_accuracy.result()*100))
    


