import torch
import torch.nn as nn
import numpy as np

from collections import defaultdict, OrderedDict

class MLP(nn.Module):
    def __init__(self, num_in, hidden_channel, num_out, skip_in=None, skip_pad=False, act_fn=None, last_act_fn=None, bias=True):
        super().__init__()

        if isinstance(hidden_channel, (tuple, list)):
            pass
        else:
            hidden_channel = [hidden_channel]

        layer_c = [num_in] + hidden_channel + [num_out]

        self.layer_c = layer_c

        self.skip_in = skip_in if isinstance(skip_in, (tuple, list)) else [skip_in]

        # self.skip_names = [ f"lin{i+1}" for i in self.skip_in ]
        if skip_pad:
            self.skip_names = [ f"lin{i+1}" for i in self.skip_in ]
        else:
            self.skip_names = [ f"lin{i}" for i in self.skip_in ]
        act_fn = (lambda : nn.ReLU()) if act_fn is None else act_fn

        for i, (ci, co) in enumerate(zip(
            layer_c[:-1],
            layer_c[1:]
            )):
            if skip_pad:
                if i-1 in self.skip_in:
                    ci = ci + layer_c[0]
            else:
                if i+1 in self.skip_in:
                    co = co - layer_c[0]

            linear = nn.Linear(ci, co, bias=bias)

            setattr(self, f"lin{i}", linear)
        
        self.activation      = act_fn()
        self.last_activation = last_act_fn() if last_act_fn is not None else nn.Identity()

        self.args_str = f"num_in={num_in},hidden_channel={hidden_channel},num_out={num_out}"
    
    def to_tcnn(self):
        # check hidden
        hidden = self.layer_c[1:-1]
        assert len(np.unique(hidden)) == 1, f"failed to convert TCNN, hidden channels are {self.layer_c[1:-1]}"
        hidden = int(np.unique(hidden)[0])

        # check skip
        assert len(self.skip_names) == 0, f"failed to convert TCNN, got skip connection {self.skip_in}"

        import tinycudann as tcnn
        act_mapping = {
            nn.ReLU:     "ReLU",
            nn.Sigmoid:  "Sigmoid",
            nn.Identity: "None"
        }
        activation = act_mapping.get(type(self.activation))
        output_act = act_mapping.get(type(self.last_activation))
        otype      = "FullyFusedMLP" if hidden in [16, 32, 64, 128] else "CutlassMLP"
        new_net    = tcnn.Network(
                        n_input_dims=self.layer_c[0],
                        n_output_dims=self.layer_c[-1],
                        network_config={
                            "otype":             otype,
                            "activation":        activation,
                            "output_activation": output_act,
                            "n_neurons":         hidden,
                            "n_hidden_layers":   len(self.layer_c)-2,
                        }
                    )
        return new_net

    def forward(self, x):
        init_x = x
        h      = x

        for i in range(len(self.layer_c)-1):
            n = f"lin{i}"
            l = getattr(self, n)
            # if i-1 in self.skip_in:
            #     x = torch.cat([init_x, x], dim=-1) / np.sqrt(2)
            if n in self.skip_names:
                x = torch.cat([x, init_x], dim=-1) * np.sqrt(0.5)
            x = l(x)
            if i != len(self.layer_c) - 2:
                x = self.activation(x)
                h = x

        x = self.last_activation(x)
        return x
