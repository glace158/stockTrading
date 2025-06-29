# RichDog
```
pip install PySide6
pip install pycryptodome
pip install GPUtil
pip install py-cpuinfo
pip install pyts
pip install PyWavelets
```

if you have this error
```
/usr/lib/python3/dist-packages/requests/init.py:87: RequestsDependencyWarning: urllib3 (2.5.0) or chardet (4.0.0) doesn't match a supported version!
warnings.warn("urllib3 ({}) or chardet ({}) doesn't match a supported "
qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "" even though it was found.
This application failed to start because no Qt platform plugin could be initialized. Reinstalling the application may fix this problem.

Available platform plugins are: minimal, vkkhrdisplay, minimalegl, offscreen, wayland, eglfs, wayland-egl, vnc, xcb, linuxfb.

Aborted (core dumped)
```

```
sudo apt-get update
sudo apt-get install libxcb-cursor0
```