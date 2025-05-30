import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# TODO remove hardcoding
TASK_AUGMENTATION = {
    "FIRE_WIRE_5CAM": [
        "In the image input, the last image is used as a reference image, with the target point being the liquid surface that the platinum wire needs to dip into. Using the right robotic arm, hold the platinum wire and carefully extend it into the outer flame of the Bunsen burner until it glows red-hot.",
        "In the image input, the last image is used as a reference image, with the target point being the liquid surface that the platinum wire needs to dip into. With the right arm, securely grasp the platinum wire and carefully extend it into the outer flame of the Bunsen burner until it turns red-hot.",
        "In the image input, the last image is used as a reference image, with the target point being the liquid surface that the platinum wire needs to dip into. Operate the right arm to carefully grasp the platinum wire and ensure it is securely held before extending it into the outer flame of the Bunsen burner until it glows red-hot.",
        "In the image input, the last image is used as a reference image, with the target point being the liquid surface that the platinum wire needs to dip into. Using the right robotic arm, pick up the platinum wire with a gentle but firm grip and extend it into the outer flame of the Bunsen burner until it turns red-hot.",
        "In the image input, the last image is used as a reference image, with the target point being the liquid surface that the platinum wire needs to dip into. With the right arm, carefully reach for the platinum wire, grasp it securely, and extend it carefully into the outer flame of the Bunsen burner until it glows red-hot."
    ],
    "FIRE_WIRE": [
        "Using the right robotic arm, hold the platinum wire and carefully extend it into the outer flame of the Bunsen burner until it glows red-hot.",
        "With the right arm, securely grasp the platinum wire and carefully extend it into the outer flame of the Bunsen burner until it turns red-hot.",
        "Operate the right arm to carefully grasp the platinum wire and ensure it is securely held before extending it into the outer flame of the Bunsen burner until it glows red-hot.",
        "Using the right robotic arm, pick up the platinum wire with a gentle but firm grip and extend it into the outer flame of the Bunsen burner until it turns red-hot.",
        "With the right arm, carefully reach for the platinum wire, grasp it securely, and extend it carefully into the outer flame of the Bunsen burner until it glows red-hot."
    ],
    "DIP_CUSO4":[
        "Using the right robotic arm, carefully hold the platinum wire and gently insert it into the bottle to dip in the CuSO4 solution.",
        "With the right arm, securely grasp the platinum wire and carefully extend it into the bottle to dip in the CuSO4 solution.",
        "Operate the right arm to carefully grasp the platinum wire and ensure it is securely held before inserting it into the bottle to dip in the CuSO4 solution.",
        "Using the right robotic arm, pick up the platinum wire with a gentle but firm grip and insert it into the bottle to dip in the CuSO4 solution.",
        "With the right arm, carefully reach for the platinum wire, grasp it securely, and insert it carefully into the bottle to dip in the CuSO4 solution."
    ],
    "PICK_UP_GLASS_ROD": [
        "In the image input, the last image serves as the reference image for picking up the glass rod, where the red point is the target point for the robotic arm to grasp.Using the right arm of the robotic arm, carefully grasp the glass rod and lift it gently.",
        "In the image input, the last image serves as the reference image for picking up the glass rod, where the red point is the target point for the robotic arm to grasp.With the right arm, securely pick up the glass rod and hold it steady.",
        "In the image input, the last image serves as the reference image for picking up the glass rod, where the red point is the target point for the robotic arm to grasp.Operate the right arm to carefully grasp the glass rod and ensure it is securely held.",
        "In the image input, the last image serves as the reference image for picking up the glass rod, where the red point is the target point for the robotic arm to grasp.Using the right robotic arm, pick up the glass rod with a gentle but firm grip.",
        "In the image input, the last image serves as the reference image for picking up the glass rod, where the red point is the target point for the robotic arm to grasp.With the right arm, carefully reach for the glass rod, grasp it securely, and lift it carefully.",
    ],
    "PICK_UP_GLASS_ROD_3CAM": [
        "Using the right arm of the robotic arm, carefully grasp the glass rod and lift it gently.",
        "With the right arm, securely pick up the glass rod and hold it steady.",
        "Operate the right arm to carefully grasp the glass rod and ensure it is securely held.",
        "Using the right robotic arm, pick up the glass rod with a gentle but firm grip.",
        "With the right arm, carefully reach for the glass rod, grasp it securely, and lift it carefully.",
    ],
    "PICK_BOTTLE_BLACK":[
        "Using the right robotic arm, carefully grasp the bottle and place it gently into the box.",
        "With the right arm, securely pick up the bottle and hold it steady before placing it into the box.",
        "Operate the right arm to carefully grasp the bottle and ensure it is securely held before putting it into the box.",
        "Using the right robotic arm, pick up the bottle with a gentle but firm grip and place it into the box.",
        "With the right arm, carefully reach for the bottle, grasp it securely, and place it carefully into the box.",
    ],
    "PLUG_DUAL_PROOG_CHARGER":[
        "With the dual-prong charger held in the right robotic arm, carefully align it with the power strip on the table and plug it in securely.",
        "Using the right robotic arm, precisely position the dual-prong charger into the power strip on the table, ensuring a firm connection.",
        "With the right robotic arm, carefully guide the dual-prong charger into the power strip on the table and plug it in securely.",
        "Using the right robotic arm, gently but firmly insert the dual-prong charger into the power strip on the table.",
        "With the dual-prong charger in the right robotic arm, carefully align and plug it into the power strip on the table.",
    ],
    "PLUG_USB":[
        "With the USB charger held in the right robotic arm, carefully align the USB end with the USB port on the device and plug it in securely."
        "Using the right robotic arm, precisely position the USB charger into the USB port on the device, ensuring a firm connection."
        "With the right robotic arm, carefully guide the USB charger into the USB port on the device and plug it in securely."
        "Using the right robotic arm, gently but firmly insert the USB charger into the USB port on the device."
        "With the USB charger in the right robotic arm, carefully align and plug it into the USB port on the device."
    ],
    "STACK_BLOCK": [
        "Use left and right arm to stack the three blocks in the red rectangle.",
        "Stack the three blocks on top of each other in the red square with dual arms.",
    ],
}

HALT_COMMANDS = [
    "halt",
    "stop moving",
    "hold still",
]


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class AirbotInputs(transforms.DataTransformFn):
    """Inputs for the Airbot policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width].
    - state: [14]
    - actions: [action_horizon, 14]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    model_type: _model.ModelType = _model.ModelType.PI0

    # Whether to randomly choose prompt in TASK_AUGMENTATION, otherwise use the first one.
    prompt_augmentation: bool = False

    # Probability replace the action with state, and replace the prompt with HALT_COMMANDS.
    halt_injection_prob: float = 0.0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0

        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]

        # Assume that base image always exists.
        base_image = _parse_image(in_images["cam_high"])

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
            "tp_0_rgb":"cam_tp",
            "ref_0_rgb": "cam_ref"
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = _parse_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                raise ValueError(f"source image {source} not found in {in_images}")
                # TODO add right_only support
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_ if mask_padding else np.True_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        # if "prompt" in data: # and "actions" in data:
        #     if data["prompt"] not in TASK_AUGMENTATION:
        #         raise ValueError(f"prompt should be keys of TASK_AUGMENTATION, got: {data['prompt']}")
        #     if self.prompt_augmentation:
        #         inputs["prompt"] = np.random.choice(TASK_AUGMENTATION[data["prompt"]])
        #     else:
        #         inputs["prompt"] = TASK_AUGMENTATION[data["prompt"]][0]
        # else:
        #     # inputs["prompt"] = "Plugging the dual-prong charger into the power strip on the table."
        inputs["prompt"] = "In the image input, the last image serves as the reference image for picking up the glass rod, where the red point is the target point for the robotic arm to grasp.Using the right arm of the robotic arm, carefully grasp the glass rod and lift it gently."

        if "actions" in data and np.random.uniform() < self.halt_injection_prob:
            inputs["prompt"] = np.random.choice(HALT_COMMANDS)
            inputs["actions"][:] = state

        return inputs


@dataclasses.dataclass(frozen=True)
class AirbotOutputs(transforms.DataTransformFn):
    """Outputs for the Airbot policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 14 dims.
        actions = np.asarray(data["actions"][:, :14])
        return {"actions": actions}
