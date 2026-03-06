"""Allow running as: python -m video_generator"""
from .cli import main
import sys

sys.exit(main())
