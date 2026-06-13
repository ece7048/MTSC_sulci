#Author: Michail Mamalakis
#Version: 0.1
#Licence:

import sys as _sys

from .pre_training import pretraining
from .fine_tuning import fine_tuning
from .scripts import classification
from .utilities import (
    CNN3D,
    DiffModel,
    SwiftUnet3D,
    config,
    create_3Dnet,
    load_data,
    monai_utils,
    preprocessing,
    trainer,
    trainer_class,
    trainer_lit,
    transformation_utils,
)

_sys.modules.setdefault(f"{__name__}.pretraining", pretraining)
_sys.modules.setdefault(f"{__name__}.classification", classification)
_sys.modules.setdefault(f"{__name__}.CNN3D", CNN3D)
_sys.modules.setdefault(f"{__name__}.DiffModel", DiffModel)
_sys.modules.setdefault(f"{__name__}.SwiftUnet3D", SwiftUnet3D)
_sys.modules.setdefault(f"{__name__}.config", config)
_sys.modules.setdefault(f"{__name__}.create_3Dnet", create_3Dnet)
_sys.modules.setdefault(f"{__name__}.load_data", load_data)
_sys.modules.setdefault(f"{__name__}.monai_utils", monai_utils)
_sys.modules.setdefault(f"{__name__}.preprocessing", preprocessing)
_sys.modules.setdefault(f"{__name__}.trainer", trainer)
_sys.modules.setdefault(f"{__name__}.trainer_class", trainer_class)
_sys.modules.setdefault(f"{__name__}.trainer_lit", trainer_lit)
_sys.modules.setdefault(f"{__name__}.transformation_utils", transformation_utils)
