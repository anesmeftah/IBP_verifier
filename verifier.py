import onnx 
from onnx import numpy_helper
import numpy as np



""" 
For our work , we will suppose for now that the X is one dimensional array. 
This will make the maths and the implementation easier as a first version implementation.
"""

class IBPVerifier:
    def __init__(self , model = None , weights = None , bias = None):
        """initiate the Verifier. Requires .onnx model path or input manual weights and bias for one layer"""

        if model is not None:
            self.layers = self.extract_layers(model)

        elif weights is not None:
            if bias is None:
                bias = np.zeros(weights.shape[0])

            else:
                if weights.shape[0] != bias.shape[0]:
                    ValueError("Weights and bias don't have compatible shapes")


            self.layers = [{
                "type" : "Linear",
                "weights" : np.asarray(weights),
                "bias" : np.asarray(bias)
            }]

            
                
        else:
            raise ValueError("Provide either model path or weights")

                


    def extract_layers(self , onnx_path):
        """
        Extract layers and parameters from the model.
        """
        
        model = onnx.load(onnx_path)
        graph = model.graph

        initializers = {}
        for init in graph.initializer:
            initializers[init.name] = numpy_helper.to_array(init)


        layers = []

        for node in graph.node:
            if node.op_type == "Gemm":
                W = initializers[node.input[1]]

                if len(node.input) > 2:
                    b = initializers[node.input[2]]
                else:
                    b = np.zeros(W.shape[0])

                layers.append({
                "type": "Linear",
                "weights": W,
                "bias": b
                })


            elif node.op_type == "Relu":

                layers.append({
                "type": "ReLU"
            })


        return layers



    def propagate_interval(self ,lower , upper , layer = {"type" : "Linear"}):
        """Vectorized Interval Calculator"""


        # Linear Layer
        if layer["type"] == "Linear":
            weights = layer["weights"]
            bias = layer["bias"]

            # Seperate positive and negative weight components
            w_pos = np.maximum(weights , 0)
            w_neg = np.minimum(weights , 0)

            # for debug
            # print("w_pos : " , w_pos.shape)
            # print("w_neg : " , w_neg.shape)
            # print("upper : " , upper.shape)
            # print("lower : " , lower.shape)

            z_upper = w_pos @ upper + w_neg @ lower + bias
            z_lower = w_pos @ lower + w_neg @ upper + bias

            print("z_upper : " , z_upper)
            print("z_lower : " , z_lower)

        # ReLU Layer
        elif layer["type"] == "ReLU":
            z_upper = np.maximum(0, upper)
            z_lower = np.maximum(0, lower)

        return z_lower , z_upper

    def forward(self , x , eps):
        lower = x - eps
        upper = x + eps

        print("lower : " , lower)
        print("upper : " , upper)


        for layer in self.layers:
            lower , upper = self.propagate_interval(lower , upper , layer)


        return lower , upper



