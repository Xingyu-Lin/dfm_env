from matplotlib import pyplot as plt
import numpy as np
from dm_control import mujoco

simple_MJCF = """
<mujoco>
<worldbody>
<light name ="top" pos ="0 0 1.5"/>
<geom name ="floor" type ="plane" size ="1 1 .1"/>
<body name ="box" pos ="0 0 .3" >
<joint name ="up_down" type ="slide" axis ="0 0 1"/>
<geom name ="box" type ="box" size =".2 .2 .2" rgba ="1 0 0 1"/>
<geom name ="sphere" pos =".2 .2 .2" size =".1" rgba ="0 1 0 1"/>
</body>
</worldbody>
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(simple_MJCF)
pixels = physics.render(height=240, width=320)
print(pixels)
plt.imshow(pixels)
# plt.savefig('./test.png')
