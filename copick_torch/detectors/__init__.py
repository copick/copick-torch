"""
Particle detectors for CryoET data.
"""

from copick_torch.detectors.monai_detector import MONAIParticleDetector
from copick_torch.detectors.dog_detector import DoGParticleDetector

__all__ = ["MONAIParticleDetector", "DoGParticleDetector"]
