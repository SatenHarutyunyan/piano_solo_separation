from .BaseNN import *
import tensorflow as tf


class UNET(BaseNN):

    def network(self, X):
        X = tf.expand_dims(X[:, :320, 1:], -1, name="X_expanded")
        train_batch_size= tf.shape(X)[0]
        class conlayer_left():
            def __init__(self, ker, in_c, out_c, name=''):
                self.w = tf.Variable(tf.random_normal([ker, ker, in_c, out_c], stddev=0.05))
                self.name = name

            def feedforward(self, input, stride=1, activation="relu"):
                self.input = input
                self.layer = tf.nn.conv2d(input, self.w, strides=[1, stride, stride, 1], padding='SAME') #make stride one horizontal and vertical because we have a rectangle
                if activation == "relu":
                    self.layerA = tf.nn.relu(self.layer, name=self.name)
                elif activation == "sigmoid":
                    self.layerA = tf.nn.sigmoid(self.layer, name=self.name)
                return self.layerA

        class conlayer_right():

            def __init__(self, ker, in_c, out_c, name=''):
                self.w = tf.Variable(tf.random_normal([ker, ker, out_c, in_c], stddev=0.05))
                self.name = name

            def feedforward(self, input, stride=1, dilate=1, output=1):
                self.input = input
                current_shape_size = input.shape
                # print("current_shape_size: ",current_shape_size)
                self.layer = tf.nn.conv2d_transpose(input, self.w,
                                                    output_shape=[train_batch_size] + [
                                                        int(current_shape_size[1].value * 2),
                                                        int(current_shape_size[2].value * 2),
                                                        int(current_shape_size[3].value / 2)],
                                                    strides=[1, 2, 2, 1], padding='SAME')
                self.layerA = tf.nn.relu(self.layer, name=self.name)
                return self.layerA


        layer1_1 = conlayer_left(3, 1, 64, "1.1").feedforward(X)
        layer1_2 = conlayer_left(3, 64, 64, '1.2').feedforward(layer1_1)

        layer2_input = tf.nn.max_pool(layer1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',name="2_input")
        layer2_1 = conlayer_left(3, 64, 128, "2.1").feedforward(layer2_input)
        layer2_2 = conlayer_left(3, 128, 128, "2.2").feedforward(layer2_1)

        layer3_input = tf.nn.max_pool(layer2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',name="3_input")
        layer3_1 = conlayer_left(3, 128, 256, '2.4').feedforward(layer3_input)
        layer3_2 = conlayer_left(3, 256, 256, '3.1').feedforward(layer3_1)

        layer4_input = tf.nn.max_pool(layer3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID',name = "4_input")
        layer4_1 = conlayer_left(3, 256, 512, '4.1').feedforward(layer4_input)
        layer4_2 = conlayer_left(3, 512, 512, '4.2').feedforward(layer4_1)

        layer5_input = tf.nn.max_pool(layer4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name = "5_input")
        layer5_1 = conlayer_left(3, 512, 1024, '5.1').feedforward(layer5_input)
        layer5_2 = conlayer_left(3, 1024, 1024, '5.2').feedforward(layer5_1) # [?, 20,15,1024]

        layer6_half_input = conlayer_right(2, 1024, 512, '6_half_input').feedforward(layer5_2) #here dimensionfirst dimension becoms a number
        layer6_input = tf.concat([layer6_half_input, layer4_2], axis=3, name="6_input")
        layer6_1 = conlayer_left(3, 1024, 512,'6.1').feedforward(layer6_input)
        layer6_2 = conlayer_left(3, 512, 512,'6.2').feedforward(layer6_1)

        layer7_half_input = conlayer_right(2,512,256,"7_half_input").feedforward(layer6_2)
        layer7_input = tf.concat([layer7_half_input, layer3_2], axis=3,name="7_input")
        layer7_1 = conlayer_left(3,512,256,"7.1").feedforward(layer7_input)
        layer7_2 = conlayer_left(3, 256, 256,'7.2').feedforward(layer7_1)

        layer8_half_input = conlayer_right(2,256,128,"8_half_input").feedforward(layer7_2)
        layer8_input = tf.concat([layer8_half_input, layer2_2], axis=3, name="8_input")
        layer8_1 = conlayer_left(3, 256, 128,'8.1').feedforward(layer8_input)
        layer8_2 = conlayer_left(3, 128, 128,'8.2').feedforward(layer8_1)

        layer9_half_input = conlayer_right(2,128,64, "9_half_input").feedforward(layer8_2)
        layer9_input= tf.concat([layer9_half_input, layer1_2], axis=3,name="9_input")
        layer9_1 = conlayer_left(3, 128, 64,'9.1').feedforward(layer9_input)
        layer9_2 = conlayer_left(3, 64, 64,'9.2').feedforward(layer9_1)
        layer10 = conlayer_left(3, 64, 1,"10").feedforward(layer9_2, activation="sigmoid")
        prediction = tf.squeeze(layer10,-1,name="prediction")
        # print("prediction: ",prediction)
        # print("X: ", X)
        # print(layer1_1, "\n", layer1_2, "\n")
        # print(layer2_input, "\n", layer2_2, "\n")
        # print(layer3_input, "\n", layer3_2, "\n")
        # print(layer4_input, "\n", layer4_2, "\n")
        # print(layer5_input,"\n",layer5_2,"\n")
        # print(layer6_input, "\n", layer6_2, "\n")
        # print("concatinating: ", layer7_half_input, " and \n", layer3_2, " resulting in into 7_input", layer7_input)
        # print(layer7_input, "\n", layer7_1, "\n", layer7_2, "\n")  # , layer7_3)
        # print(layer8_half_input, '\n', layer2_2, layer8_input)
        # print(layer8_input, "\n", layer8_1, "\n", layer8_2)
        # print(layer9_input, "\n", layer9_1, "\n", layer9_2, "\n")
        # print(layer10)
        return prediction

    def loss(self, y, y_pred):
        y = y[:, :320, 1:]
        print("loss working")
        print("y is: ",y)
        print("y pred :", y_pred)
        l2_loss = tf.reduce_mean(tf.square(y - y_pred))  # originaly used doftmax with cross entropy
        tf.summary.scalar("Loss", l2_loss)
        return l2_loss