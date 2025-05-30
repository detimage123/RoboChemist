import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    ALOHA_PICK_GLASS_ROD = "pi0_pick_glass_rod3_400_5cam"
    ALOHA_FIRE_WIRE = "pi0_fire_wire_ych3_5cam"
    ALOHA_DIP_CUSO4 = "pi0_dip_cuso43_5cam"
    ALOHA_PRESS_HEATER = "pi0_press_heater"
    ALOHA_FIRE_TEST_TUBE = "pi0_fire_test_tube"
    ALOHA_STIR_SOLUTION = "pi0_stir_solution"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi0_aloha",
        dir="s3://openpi-assets/checkpoints/pi0_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="s3://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi0_fast_droid",
        dir="s3://openpi-assets/checkpoints/pi0_fast_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi0_fast_libero",
        dir="s3://openpi-assets/checkpoints/pi0_fast_libero",
    ),
    EnvMode.ALOHA_PICK_GLASS_ROD: Checkpoint(
        config="pi0_pick_up_glass_rod3_400_5cam",
        dir="/home/agilex/ych_ckpt/pi0_pick_up_glass_rod3_400_5cam/30000"
    ),
    EnvMode.ALOHA_FIRE_WIRE: Checkpoint(
        config="pi0_fire_wire_ych3_5cam",
        dir="/home/agilex/ych_ckpt/pi0_fire_wire_ych3_5cam/29999"
    ),
    EnvMode.ALOHA_DIP_CUSO4: Checkpoint(
        config="pi0_dip_cuso43_5cam",
        dir="/home/agilex/ych_ckpt/pi0_dip_cuso43_5cam/29999"
    ),
    EnvMode.ALOHA_STIR_SOLUTION: Checkpoint(
        config="pi0_stir_solution_5cam",
        dir="/home/agilex/ych_ckpt/pi0_stir_solution_5cam/29999"
    ),
    EnvMode.ALOHA_PRESS_HEATER: Checkpoint(
        config="pi0_press_heater_5cam",
        dir="/home/agilex/ych_ckpt/pi0_press_heater_5cam/29999"
    ),
    EnvMode.ALOHA_FIRE_TEST_TUBE: Checkpoint(
        config="pi0_fire_test_tube_5cam",
        dir="/home/agilex/ych_ckpt/pi0_fire_test_tube_5cam/29999"
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
