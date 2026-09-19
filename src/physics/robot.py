'''
Base robot class for like, getting robot info. list of sensors, motors, etc.
Should be extended for specific implementations, such as the actual hand 
versions and variants.

Maybe some physics data lives here too? like the robot joints and stuff?
or maybe that all lives more generally within the simulation and this layer
is just in charge of storing data on like the specific robot design? idk
'''
class Robot:
    def __init__(self):
        self.sensors = []
        self.actuators = []
        self.position = [0, 0, 0]
        self.orientation = [0, 0, 0, 1]
    
    def add_actuator(self, actuator):
        self.actuators.append(actuator)
