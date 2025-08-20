"""
TriadFuse: Compound adversarial attack framework for document VLMs
"""

__version__ = "0.1.0"

from .eot import EOT
from .heads.texture import TextureHead
from .surrogate.base import SurrogateModel
from .surrogate.donut import DonutSurrogate

__all__ = [
    "EOT",
    "TextureHead",
    "SurrogateModel",
    "DonutSurrogate",
]
