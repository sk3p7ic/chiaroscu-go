package mlp

import (
    "math/rand"
    "gonum.org/v1/gonum/mat"
)

// A model--essentially a collection of layers.
type Model struct {
    Layers []*Layer
    NClasses uint
}

// Forward pass through the model for a given batch of inputs. Returns the
// output of the model.
func (m *Model) Forward(x *mat.Dense) *mat.Dense {
    for _, layer := range m.Layers[1:] { // Skip the input layer.
        x = layer.Forward(x)
    }
    return x
}

// Backward pass through the model for a given batch of inputs and outputs. The
// model is updated in place. Note that this function assumes that the model
// only has three layers (input, hidden, output).
func (m *Model) BackwardThreeLayer(y *mat.Dense) {
    if len(m.Layers) != 3 {
        panic("BackwardThreeLayer only works for models with three layers.")
    }
    Y := OneHotEncodeOnVector(y, m.NClasses)
    dZ2 := m.Layers[2].LastOutput
    dZ2.Sub(dZ2, Y)
    r, c := m.Layers[2].Weights.Dims()
    dW2 := mat.NewDense(r, c, nil)
    dW2.Mul(dZ2, m.Layers[1].LastOutput.T())
    dW2.Scale(1/float64(r), dW2)
}

// Train the model on a given batch of inputs and outputs. The number of epochs
// specifies how many times to run through the training data, and lr is the
// learning rate. Note that this function does not shuffle the data, so it is
// assumed that the data is already shuffled. Furthermore, there is no batching
// implemented yet, so the entire dataset is used for each epoch.
func (m *Model) Train(x, y *mat.Dense, epochs uint, lr float64) {
    // TODO
}

// Initialize a new model with the given layer sizes and activation functions.
func NewModel(sizes []LSize, activations []ActivationFunction) *Model {
    rand.Seed(0)
    layers := make([]*Layer, len(sizes))
    for i, size := range sizes {
        layers[i] = NewLayer(int(size.InputWidth), int(size.InputHeight),
            int(size.OutputWidth), int(size.OutputHeight), activations[i])
    }
    return &Model{Layers: layers, NClasses: 10}
}
