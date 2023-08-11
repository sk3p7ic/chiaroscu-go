package mlp

// A model--essentially a collection of layers.
type Model struct {
    Layers []*Layer
}

// Initialize a new model with the given layer sizes and activation functions.
func NewModel(sizes []LSize, activations []ActivationFunction) *Model {
    layers := make([]*Layer, len(sizes))
    for i, size := range sizes {
        layers[i] = NewLayer(int(size.Width), int(size.Height), activations[i])
    }
    return &Model{Layers: layers}
}
