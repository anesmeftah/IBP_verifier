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

    def linear(self, W, bias):
        A_new = W @ self.A
        b_new = W @ self.b + bias

        return SymbolicBound(A_new, b_new)


    def relu(self, lower, upper):
        """
        Apply ReLU relaxation on symbolic bounds.

        lower: concrete lower bounds for each neuron
        upper: concrete upper bounds for each neuron
        """

        A_new = self.A.copy()
        b_new = self.b.copy()

        for i, (l,u) in enumerate(zip(lower,upper)):

            if u <= 0:
                A_new[i,:] = 0
                b_new[i] = 0


            elif l >= 0:
                continue


            else:
                alpha = u / (u - l)
                beta = -u * l / (u - l)

                A_new[i, :] = alpha * A_new[i, :]
                b_new[i] = alpha * b_new[i] + beta

        return SymbolicBound(A_new, b_new)

    def concretize(self, x_lower, x_upper):
        """
        Compute concrete interval bounds from symbolic bounds.

        z = A*x + b

        x_lower: lower input bounds
        x_upper: upper input bounds

        returns:
            lower bound of z
            upper bound of z
        """

        A_pos = np.maximum(self.A, 0)
        A_neg = np.minimum(self.A, 0)

        lower = (
            A_pos @ x_lower +
            A_neg @ x_upper +
            self.b
        )

        upper = (
            A_pos @ x_upper +
            A_neg @ x_lower +
            self.b
        )

        return lower, upper
