package mlp

import (
    "math"
    "gonum.org/v1/gonum/mat"
)

// An activation function for a layer in the neural network.
type ActivationFunction func(*mat.Dense) *mat.Dense

// Performs the ReLU function on the given matrix element-wise in which each
// element is passed through the function f(x) = max(0, x).
func ReLU(m *mat.Dense) *mat.Dense {
    // Apply the ReLU function to each element in the matrix.
    f := func(_, _ int, v float64) float64 {
        return math.Max(0, v)
    }
    r, c := m.Dims()
    ret := mat.NewDense(r, c, nil)
    ret.Apply(f, m)
    return ret
}

// Performs the softmax function on the given matrix element-wise in which each
// element is passed through the function f(x, m) = e^x / sum(e^m) where m is
// the matrix.
func Softmax(m *mat.Dense) *mat.Dense {
    // Apply the softmax function to each element in the matrix.
    f := func(_, _ int, v float64) float64 {
        return math.Exp(v)
    }
    r, c := m.Dims()
    ret := mat.NewDense(r, c, nil)
    ret.Apply(f, m)
    sum := mat.Sum(ret)
    ret.Scale(1/sum, ret)
    return ret
}
