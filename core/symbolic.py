import numpy as np

class SymbolicBound:
    """Represent affine bounds : A * x + b where x is the input"""
    def __init__(self , A , b):
        self.A = A
        self.b = b

    @classmethod
    def from_input(cls , input_dim):
        """Create symbolic representation of the input"""
        A = np.eye(input_dim)
        b = np.zeros(input_dim)

        return cls(A,b)


    def __repr__(self):
        return (
            f"SymbolicBound(\n"
            f"A = \n{self.A}\n"
            f"b = \n{self.b}\n"
        )

    