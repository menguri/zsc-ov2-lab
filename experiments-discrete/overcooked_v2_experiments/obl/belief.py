from flax import nn
import distrax
from jaxmarl.environments.overcooked_v2.overcooked import State
from jaxmarl.environments.overcooked_v2.layouts import Layout


class ItemDistribution(nn.Module):
    num_states: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_states)(x)

        return distrax.Categorical(logits=x)


class StateDistribution(nn.Module):
    layout: Layout

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.layout.num_states)(x)

        return distrax.Categorical(logits=x)

class AgentDistribution(nn.Module):
    layout: Layout

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(2)(x)

        return distrax.Categorical(logits=x)
        


class BeliefModel(nn.module):
    features: int
    layout: Layout

    @nn.compact
    def __call__(self, x):
        obs, agent_id, hstate = x

        encoder = nn.RNN(
            nn.LSTMCell(self.hidden_size), return_carry=True, name="encoder"
        )

        ff = nn.MLP(self.features)(x)





        return next_hstate, state_belief
