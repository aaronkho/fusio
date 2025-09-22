import platform
from packaging.version import Version
__version_object__ = Version("0.2.0")
__version__ = str(__version_object__)
__python_version_object__ = Version(platform.python_version())
__python_version__ = str(__python_version_object__)
