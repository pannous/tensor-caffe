# TODO pre-alpha version 0.0000
# 100% Work in progress!!
# do not try to use yet
# see https://github.com/PrincetonVision/marvin/tree/master/tools/converter_caffe

import tensorflow as tf
import caffe_pb2  # Automatically generated protobuf parser for caffe.prototxt net graphs
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from google.protobuf import text_format


def parse_args():
    parser = ArgumentParser(description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--net_proto_file', help='Input network prototxt file', default="lenet.prototxt2")
    args = parser.parse_args()
    return args


class CaffeGraphBuilder():
    def __init__(self, *margs, **kwargs):  # Constructor
        self.layers=[]
        self.methods = dir(self)
        self.previous_layer = self.current_layer = None
        self.net = net = kwargs['net']
        print("LOADING CAFFE NET %s" % net.name)
        print("input_shape input_dim %s %s" % (net.input_shape, net.input_dim))
        print("?? input %s" % net.input)

    def import_graph(self):
        for l in self.net.layer:  # NEW
            self.add_layer(l)
            print("%s" % (l))
            # print(l.name)
            # print(l.type)
            # print(l.bottom)
            # assert l.bottom==self.previous_layer.name or todo
            # print(l.top) # todo

    def add_layer(self, layer):
        self.previous_layer = previous = self.current_layer
        self.current_layer = layer
        if layer.type in self.methods:
            print("KNOWN type: %s"% layer.type)
            method = getattr(self, layer.type)
            method(previous,layer)
        else:
            raise "UNKNOWN type: %s !!!"% layer.type
            # print("Unknown type: %s !!!"% layer.type)


    # p previous l layer
    def Convolution(self,previous,layer):
        input=self.previous_layer
        strides=[1,2,2,1]
        filter=[1,2,2,1]
        padding='SAME'
        # padding=[1,2,2,1]
        self.layers+=tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)

    def InnerProduct(self,previous,layer):
        a=previous
        b=self.current_layer
        self.layers+=tf.nn.math_ops.matmul(a, b,
           transpose_a=False, transpose_b=False,
           a_is_sparse=False, b_is_sparse=False,
           name=None)

    def Softmax(self,previous,layer):
        logits = previous
        self.layers+=tf.nn.softmax(logits, name=None)

    def Pooling(self,previous,layer):
        value = previous
        ksize = [1,2,2,1]
        strides = [1,2,2,1]
        padding='SAME'
        self.layers+=tf.nn.avg_pool(value, ksize, strides, padding, name=None)
        self.layers+=tf.nn.max_pool(value, ksize, strides, padding, name=None)

    def ReLU(self,previous,layer):
        features = previous
        self.layers+=tf.nn.relu(features, name=None)



def main():
    args = parse_args()
    net = caffe_pb2.NetParameter()
    text_format.Merge(open(args.net_proto_file).read(), net)
    builder = CaffeGraphBuilder(net=net)
    builder.import_graph()


if __name__ == '__main__':
    main()
