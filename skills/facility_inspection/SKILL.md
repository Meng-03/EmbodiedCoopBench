---
name: facility_inspection
description: "Standard facility inspection and energy audit checklist. Trigger this skill upon receiving a facility inspection request for a designated room. Based on the requirements of each checklist item and the real-time observed state of the facilities, dynamically allocate robots to execute the appropriate semantic action primitives."
---

## Facility Inspection Checklist
### Overview
Standard facility inspection and energy audit checklist. Trigger this skill upon receiving a facility inspection request for a designated room. Based on the requirements of each checklist item and the real-time observed state of the facilities, dynamically allocate robots to execute the appropriate semantic action primitives.

### Facility Inspection
#### 1. Laptop Closure Verification
- workflow: Inspect the state of the laptop. If it is observed to be open, the robot need to close the laptop.
- Involved Objects: laptop
- Required Semantic Action Primitives: navigate_to, close
#### 2. Television Power Management
- workflow: Check the operational status of the television. If it is powered on, the robot must first locate and pickup the remotecontrol from the sofa, use it to toggle off the television, and subsequently put the remotecontrol back onto the sofa.
- Involved Objects: television, remotecontrol, sofa
- Required Semantic Action Primitives: navigate_to, pickup, put_in, toggle_off
#### 3. DeskLamp Illumination Audit
- workflow: Verify the state of the desklamp. If it remains powered on, the robot need to toggle it off to conserve energy.
- Involved Objects: desklamp
- Required Semantic Action Primitives: navigate_to, toggle_off
#### 4. FloorLamp Illumination Audit
- workflow: Verify the state of the floorlamp. If it remains powered on, the robot need to toggle it off.
- Involved Objects: floorlamp
- Required Semantic Action Primitives: navigate_to, toggle_off
#### 5. Main Lighting Audit
- workflow: Inspect the room's primary lightswitch. If the switch is on, the robot need to toggle it off.
- Involved Objects: lightswitch
- Required Semantic Action Primitives: navigate_to, toggle_off


### Robot Allocation Rules
- Assign robots to actions based on the action capabilities of each robot.
- If an inspecting robot detects a device state that exceeds its own execution capabilities, it must dispatch other specialized robots to cooperate and take over the execution.
- Maximize inspection efficiency by enabling multi-robot coordination where necessary, and by continuously dispatching idle robots to address unfinished checklist items.
- Follow the logical order of actions and do not skip required steps.
- A robot must explicitly perform a navigate_to action to approach a target device in order to reliably observe its real-time status before any interaction can occur.