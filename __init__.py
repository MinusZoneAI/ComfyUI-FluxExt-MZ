

import json
import os
import sys
from nodes import MAX_RESOLUTION
import comfy.utils
import shutil
import comfy.samplers
import folder_paths


WEB_DIRECTORY = "./web"

AUTHOR_NAME = u"MinusZone"
CATEGORY_NAME = f"{AUTHOR_NAME} - FluxExt"


import importlib

NODE_CLASS_MAPPINGS = {
}


NODE_DISPLAY_NAME_MAPPINGS = {
}

from . import mz_fluxext_core
import importlib


class MZ_Flux1VRAM_MT_Patch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", )
        }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = f"{CATEGORY_NAME}"

    def load_unet(self, **kwargs):
        from . import mz_fluxext_core
        importlib.reload(mz_fluxext_core)
        return mz_fluxext_core.MZ_Flux1VRAM_MT_Patch_call(kwargs)


NODE_CLASS_MAPPINGS["MZ_Flux1VRAM_MT_Patch"] = MZ_Flux1VRAM_MT_Patch
NODE_DISPLAY_NAME_MAPPINGS["MZ_Flux1VRAM_MT_Patch"] = f"{AUTHOR_NAME} - Flux1VRAM_MT_Patch"
