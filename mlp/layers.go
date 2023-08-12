package mlp

import (
    "math/rand"
    "gonum.org/v1/gonum/mat"
)

// A size for a layer in the neural network.
type LSize struct {
    InputWidth uint
    InputHeight uint
    OutputWidth uint
    OutputHeight uint
}

// A layer in the neural network.
type Layer struct {
    Weights *mat.Dense
    Biases *mat.Dense
    LastOutput *mat.Dense
    Activation ActivationFunction
}

// Initialize a matrix of size r x c with random values, or zeros if zfill is
// true.
func initLayerMatrix(r, c int, zfill bool) *mat.Dense {
    elem := func() float64 {
        if zfill { return 0 } else { return rand.Float64() }
    }
    ret := mat.NewDense(r, c, nil)
    for i := 0; i < r; i++ {
        for j := 0; j < c; j++ {
            ret.Set(i, j, elem())
        }
    }
    return ret
}

// Initialize a new layer with random weights and a vector of zeros for biases.
func NewLayer(w, h, out_w, out_h int, activation ActivationFunction) *Layer {
    return &Layer{
        Weights: initLayerMatrix(w, h, false),
        Biases: initLayerMatrix(1, h, true),
        LastOutput: initLayerMatrix(out_w, out_h, true),
        Activation: activation,
    }
}

// Forward pass through the layer.
func (l *Layer) Forward(x *mat.Dense) *mat.Dense {
    z := mat.NewDense(0, 0, nil)
    z.Mul(x, l.Weights)
    z.Add(z, l.Biases)
    logits := l.Activation(z)
    l.LastOutput = logits
    return logits
}
