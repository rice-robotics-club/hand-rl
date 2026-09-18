# hand-rl
what a handy piece of software

hand-rl is the repository for the 2026-2027 Rice Robotics hand reinforcement learning project. 

The overall goal of this project is to create a rl-trained hand that can perform both generalized hand movement, and a specialized subnet that is capable of playing a real piano (with a heavier focus on the piano, since that's the cool part)


# general rough outline
## Physics Sim
Our physics sim is written as a layer on top of the Genesis-World library, which provides a nice interface for both running a sim agnostic to and consistent across different dev hardware and getting all the info we need to train the RL model

- We want a base environment class that just like creates an empty 3d space to put things in 
  - We can decorate this for fun. last year we made the catbot environment look like minecraft. would love to hear ideas
- we want a generalized "hand" object that is sort of abstracted from the specific hand model we're using (so we can swap out designs as our engineering team iterates on it)
  - the specific hand designs should provide info on like, the sort of 'skeleton' of the hand in terms of where its joints and connecting parts live in a minimal physics sense and then where its 3d model parts live and should be attached to that skeleton in the sim.
    -  this is a pretty common thing in robotics so theres a few differerent common file formats like .xml and .urdf that are used for this that genesis should be able to work with pretty plug-and-play
- we want a piano object that can be placed in the environment and interact with the hand.
  - needs to have like methods to get the positions of the keys boundary boxes similar to what we want the PianoMind neural net to do (so we can easily check its answers and coach it)
  - would be neat if it actually made sound in the sim hahaha 
    - does genesis support audio and simulated microphones??? i actually have no idea lol 

## Reinforcement Learning
Our RL model consists of one decision-making network that utilizes three specialized subnetworks: 
### GeneralPerception
- Camera input --> Object identification. Marks positions of objects in the hand's workspace, 
  - formatted as a list of object names and their bounding box in 3d space 
### MotorActor
- Planned sequence of "keyframe" positions for the hand to hit --> actual control over the hand that does so in the most energy efficient way possible / the best it can do within safe limits of the real system
- 
### PianoMind
- Camera input --> specific key positions when a piano is detected
  - formatted as like a list of key names and their bounding boxes in 3d space
- MIDI song input + external clock + key positions from the prior network --> planned sequence of hand+finger joint movements
  - formatted as a sequence of joint angles for each hand+finger at each timestep
  - This is going to need to be a convolutional neural network of some sort to read the midi input properly.

This set of tools is probably enough for us to directly control it pretty well (i.e. run some script that loads in a MIDI file and generates a plan for the hand, then feeds that to the MotorActor model) ((maybe we want a different structure that allows it to adjust in realtime if the keyboard moves under it or something? right now the plan is "baked" once and sort of set in stone after that. at the very least maybe having the ability to apply a coordinate transform to it after the fact to adjust to real piano position would be nice. worth thinking about))

## audio-tools
- responsible for the loading of a midi file and conversion into a nice array of note events that our neural network can realistically parse
- we also mentioned once that it might be neat generating note events from audio input. could be fun to just like generate a midi track from arbitrary .mp3 input... sing to the robot and it plays along or something?

# Overall Goals / Timeline

# Pure Simulation goals: 
Milestone 0: Hand is loaded in a 3d sim environment and can be controlled manually by a simple script

Milestone 1: Hand can be controlled by an RL model doing random actions

Milestone 2: Stationary hand can move fingers and play singular notes with RL

Milestone 3: Stationary hand can play chords with RL 

Milestone 4: Hand+Arm can move, play more complex sequences of notes across full keyboard with RL

Milestone 5: Hand+Arm can play through a full song with RL

# Sim2Real goals: 
Milestone 0: designated training environment for sim2real transfer learning that loads in a pure-sim trained neural network and its bundle of reward functions. 
- Could also do a student-teacher separation here, where maybe the pure-sim is a bigger more complex model with access to unrealistic kinds of data and then we start up a fresh network here in sim2real that has the actual limitations of the real robot and tries to imitate the teacher model's behavior?

Milestone 1: Random sensor noise is simulated for everything (actuators, cameras, mics, etc)

Milestone 2: Motor torque, resistance, physical parameters like that are all noisy as well 

Milestone 3: 

# Real Robot goals:
Milestone 0: some sort of very basic onboard control loop with ros2 exists on the hand

Milestone 1: Camera, Mic, joint position sensors are all connected as ros2 topics

Milestone 2: reinforcement learning models can be deployed on the robot to control hand movements

Milestone 3:  
