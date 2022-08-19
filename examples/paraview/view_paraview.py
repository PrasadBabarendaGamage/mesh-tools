from paraview.simple import *

# Create a new sphere proxy on the active connection and register it
# in the sources group.
sphere = Sphere(ThetaResolution=16, PhiResolution=32)

# Apply a shrink filter
shrink = Shrink(sphere)

# Turn the visibility of the shrink object on.
Show(shrink)

# Render the scene
Render()
a=1