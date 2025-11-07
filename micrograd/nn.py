import random
import numpy as np
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        # x could be a single sample (list/tuple of Values) or a batch (list of lists/tuples of Values)
        
        try:
            # Try to treat as single sample first
            # Compute sum(w_i * x_i) for this sample
            sum_wx = self.w[0] * x[0]
            for j in range(1, len(self.w)):
                sum_wx += self.w[j] * x[j]
            # Add bias and apply activation
            act = sum_wx + self.b
            return act.relu() if self.nonlin else act
        except (TypeError, IndexError):
            # If it fails, treat as batch input
            # Batch input: x is [batch_size, nin]
            outs = []
            for sample in x:
                # Compute sum(w_i * x_i) for this sample
                sum_wx = self.w[0] * sample[0]
                for j in range(1, len(self.w)):
                    sum_wx += self.w[j] * sample[j]
                # Add bias and apply activation
                act = sum_wx + self.b
                outs.append(act.relu() if self.nonlin else act)
            return outs

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        # x could be a single sample (list/tuple of Values) or a batch (list of lists/tuples of Values)
        
        # Get output from each neuron
        neuron_outs = [n(x) for n in self.neurons]
        
        # Check if output is a batch
        try:
            # Try to treat as single sample output
            # If neuron_outs[0] is a Value, then it's a single sample
            if isinstance(neuron_outs[0], Value):
                return neuron_outs[0] if len(neuron_outs) == 1 else neuron_outs
            else:
                # It's a batch output
                # Transpose the output to get [batch_size, nout]
                batch_size = len(neuron_outs[0])
                nout = len(neuron_outs)
                
                batch_outs = []
                for i in range(batch_size):
                    sample_out = []
                    for j in range(nout):
                        sample_out.append(neuron_outs[j][i])
                    # Return single value if only one output neuron
                    batch_outs.append(sample_out[0] if nout == 1 else sample_out)
                return batch_outs
        except (TypeError, IndexError):
            # Fallback for edge cases
            return neuron_outs[0] if len(neuron_outs) == 1 else neuron_outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
