from .abstract import ActorCriticBase
from .rnn import ScannedRNN, ActorCriticRNN
from .cnn import ActorCriticCNN
from jaxmarl.environments.overcooked_v2.common import Actions


def get_actor_critic(config) -> ActorCriticBase:
    model_config = dict(config["model"])
    for key in ("ACTION_PREDICTION", "STATE_PREDICTION", "PRED_Z_DIM"):
        if key in config:
            model_config[key] = config[key]

    match model_config["TYPE"]:
        case "RNN":
            actor_critic = ActorCriticRNN
        case "CNN":
            actor_critic = ActorCriticCNN
        case _:
            raise NotImplementedError("Only RNN and CNN models are supported.")

    return actor_critic(
        # env.action_space(env.agents[0]).n,
        len(Actions),
        config=model_config,
    )


def initialize_carry(config, batch_size: int):
    model_config = config["model"]

    if model_config["TYPE"] == "RNN":
        state_prediction = bool(config.get("STATE_PREDICTION", False))
        pred_dim = (
            int(
                config.get(
                    "PRED_Z_DIM",
                    max(1, int(model_config["GRU_HIDDEN_DIM"]) // 2),
                )
            )
            if state_prediction
            else len(Actions)
        )
        return ActorCriticRNN.initialize_carry(
            batch_size,
            model_config["GRU_HIDDEN_DIM"],
            pred_dim=pred_dim,
        )

    return None
