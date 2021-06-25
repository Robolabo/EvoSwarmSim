import os
import xml.etree.cElementTree as ET
from spike_swarm_sim.objects import WorldObject3D

class Wall(WorldObject3D):
    #TODO Meter H y W variables.
    def __init__(self, *args, height=2, width=20, **kwargs):
        self.height = height
        self.width = width
        self.resize_wall()
        super(Wall, self).__init__('tmp/wall_{}x{}x2'.format(width, height), *args, static=True,\
            controller=None, tangible=True, luminous=False, **kwargs)
    
    def resize_wall(self):
        if not os.path.isfile("spike_swarm_sim/objects/urdf/tmp/wall_{}x{}x2.urdf".format(self.width, self.height)): 
            tree = ET.parse("spike_swarm_sim/objects/urdf/wall.urdf")
            root = tree.getroot()
            # aa = root.get('link').get('link')
            root.findall(".//link/visual/geometry/box")[0].attrib['size'] = '{} {} 2'.format(self.width, self.height)
            root.findall(".//link/collision/geometry/box")[0].attrib['size'] = '{} {} 2'.format(self.width, self.height)
            tree.write(open("spike_swarm_sim/objects/urdf/tmp/wall_{}x{}x2.urdf".format(self.width, self.height), 'wb'))

    def reset(self, seed=None):
        pass