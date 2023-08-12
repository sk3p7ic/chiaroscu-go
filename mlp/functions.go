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

// Performs a one-hot encoding of the given float64 value with the given number
// of classes. Returns a column vector with n_classes rows and a 1 in the row
// corresponding to the given value and 0s in all other rows.
func OneHotEncode(y float64, n_classes uint) *mat.Dense {
    ret := mat.NewDense(int(n_classes), 1, nil)
    // Ensure that all values are 0.
    for i := 0; i < int(n_classes); i++ {
        ret.Set(i, 0, 0)
    }
    // Set the value at the given index to 1.
    ret.Set(int(y), 0, 1)
    return ret
}

func OneHotEncodeOnVector(y *mat.Dense, n_classes uint) *mat.Dense {
    r, _ := y.Dims()
    ret := mat.NewDense(int(n_classes), r, nil)
    for i := 0; i < r; i++ {
        ohe := OneHotEncode(y.At(i, 0), n_classes)
        for j := 0; j < int(n_classes); j++ {
            ret.Set(j, i, ohe.At(j, 0))
        }
    }
    return ret
}
