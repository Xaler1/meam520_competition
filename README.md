# Super Block Stacker 4000

## Overall structure

There are currently 4 main threads:
- The Main thread that is main.py. Responsible for the overall flow of the program.
All the high-level decision and planning should be here. 
- The Computer thread that is run() in computer.py. Responsible for costly math computations, fk, ik, etc.
Breaks down Tasks into Commands.
- The Executor thread that is run() in executor.py. Responsible for timing the commands
and keeping track or robot configuration.
- The Manipulator thread that is run() in manipulator.py. Responsible for interacting with the arm.
Sends movement and gripper commands to the arm. Collects observations from the camera

## Communication
The overall communication flow right now is:
```
     Main <── ┓
      |       |
      v       |
   Computer   |
      |       |
      v       |
┏─>Executor ─ ┙ 
|     |        
|     v                 
└─Manipulator--> observations
      |
      v
     ROS
```

All the communication is done using multiprocessing Queue-s. Everything is
initialized at the top of main.py with intuitive names used.

Computer accepts Task that have to specify a TaskType  
Executor accepts Command that have to specify a CommandType  
Manipulator accepts Action that have to specify an ActionType

The definitions are in the respective files.  
All communication **MUST** be sent through the Computer, otherwise the
command ordering verification will break down. There is a TaskType.BYPASS
if you want to send a Command directly to the Executor.

As well as forward communication, the Executor and Manipulator send completion messages.
The Executor will send ids of completed commands and the Manipulator will send ids of completed actions.
Note that for Manipulator this means fully completed by ROS so this can be used for proxy awaiting for actions
we want to execute synchronously. This means that ids from the Manipulator can arrive entirely out of order. 
So for progress tracking only the ids returned by the Executor should be used.  
The queue from Executor to Main is also used to send block poses right now.

Each task must have an id assigned. The computer may then break down the task into multiple commands
which will extend the ids. e.g:  
"grab" will be broken into "grab-0", "grab-1", "grab-2", "grab-3"
So keep this in mind if you want to check completion. 

## To get the observations
The manipulator thread is responsible for getting the observations from the camera.
In the main a queue called "observation_queue" is initialized and it will contain an observation.  
Note that this observation will be stale, as they are generated when the queue is detected to be empty
and **not when you request it**. So you should first take an observation and then wait for a new one to be generated.
The observation will contain the camera transform, the detections, the rbg feed, and the depth feed. 


## Running tips
At the end of every run, try not to interrupt the program, but instead press Enter and wait for
it to kill all the subprocesses, otherwise you might have orphan processes running in the background,
taking up RAM.


Redundant files (ignore for now):
- controller.py
- trajectories.py