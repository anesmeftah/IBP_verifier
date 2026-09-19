import onnx 
from onnx import numpy_helper



""" 
For our work , we will suppose for now that the X is one dimensional array. 
This will make the maths and the implementation easier as a first version implementation.
"""

class IBPVerifier:
    def __init__(self , model):
        """initiate the Verifier. Requires .onnx model path"""
        self.layers = self.extract_layers(model)


    def extract_layers(self , onnx_path):
        """
        Extract layers and parameters from the model.
        """

        model = onnx.load(onnx_path)
        graph = model.graph

        initializers = {}
        for init in graph.initializers:
            initializers[init.name] = numpy_helper.to_array(init)


        layers = []

        for node in graph.node:
            if node.op_type == "Gemm":
                W = initializers[node.input[1]]
                b = initializers[node.input[2]]

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



    def Interval(self ,lower , upper , layer):
        """Interval Calculator"""

        z_upper = 0
        z_lower = 0

        # Linear Layer
        if layer["type"] == "Linear":
            weights = layer["weights"]
            bias = layer["bias"]

            # calculate upper
            for i in range(0, len(weights)):
                if weights[i] < 0 :
                    z_upper = z_upper + weights[i] * lower + bias
                    z_lower = z_lower + weights[i] * upper + bias
                else :
                    z_upper = z_upper + weights[i] * upper + bias
                    z_lower = z_lower + weights[i] * lower + bias

        # ReLU Layer
        else:
            if lower > 0 :
                return lower , upper
            elif upper < 0 : 
                return 0 , 0
            else:
                z_upper = (2/3) * upper + 4/3

                z_lower = 0


        return z_lower , z_upper

    def forward(self , x , eps):
        lower = x - eps
        upper = x + eps

        for layer in self.layers:
            lower , upper = self.Interval(lower , upper , layer)





