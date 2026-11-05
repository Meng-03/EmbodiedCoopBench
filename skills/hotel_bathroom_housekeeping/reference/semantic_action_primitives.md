
<!-- ### References
- reference/semantic_action_primitives.md: Describes the semantic action primitives used in this skill.  -->
## Semantic Action Primitives
### navigate_to
- Purpose: Move near a target object.
- Preconditions: The target object exists in the environment. 
- Postconditions: The robot is positioned near the target object, enabling further interactions.
- Applicable when: the robot needs to approach an object in order to observe its state or perform further interaction.
### pickup
- Purpose: Pick up an object, transferring the object from the environment to the robot's control.
- Preconditions: The target object is visible in the robot’s field of view; partial occlusion is acceptable.
- Postconditions: The object is held by the robot, and the robot cannot grasp another object.
- Applicable when: The robot needs to hold a visible object for later use, relocation, or task execution.
### put_in
- Purpose: Place the object currently held by the robot onto or into a target receptacle.
- Preconditions: The robot is near the target receptacle and is currently holding the target object (Openable receptacles like cabinets must be in the open state).
- Postconditions: The robot no longer holds the object, and the object is placed onto or into the target container.
- Applicable when: the robot is already holding an object and needs to place it onto or into a visible container.
### open
- Purpose: Open an object such as a container, curtain, or book.
- Preconditions: The robot is near the object, and the object is currently closed.
- Postconditions: The object is successfully opened.
- Applicable when: The robot needs to access the inside of a container, expose a previously occluded area, or enable a subsequent interaction.
### close
- Purpose: Close an object such as a container, curtain, or book.
- Preconditions: The robot is near the object, and the object is currently open.
- Postconditions: The object is successfully closed.
- Applicable when: The robot has finished the required interaction with the object or keep the environment tidy.
### clean
- Purpose: Clean a target object or surface, such as mirror, toilet or sink.
- Preconditions: The robot is near the object, and the object is currently dirty.
- Postconditions: The target object is left in a clean state.
- Applicable when: the robot observes that an object needs cleaning.