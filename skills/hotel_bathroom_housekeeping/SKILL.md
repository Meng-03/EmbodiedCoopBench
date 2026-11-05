---
name: hotel-bathroom-housekeeping-checklist
description: "Standard hotel bathroom cleaning checklist. Trigger this skill when a guest requests bathroom cleaning. Based on each checklist item and actual observed states of the room, allocate robots to execute semantic action primitives." 
---

## Bathroom Housekeeping Checklist
### Overview
Standard hotel bathroom cleaning checklist. Trigger this skill when a guest requests bathroom cleaning. Based on each checklist item and actual observed states of the room, allocate robots to execute semantic action primitives.

### Bathroom Checklist
#### 1. Toilet area cleaned
- Workflow: If the toilet is closed, the robot must first open it, then find and pickup the scrubbrush to clean the interior of the toilet. Once the cleaning is completed, it should put the scrubbrush back on the floor and close the toilet.
- Involved Objects: toilet, scrubbrush
- Required Semantic Action Primitives: navigate_to, open, close, pickup, put_in, clean
#### 2. Used toilet paper cleared and replenished
- Workflow: Ensure the toiletpaperhanger has a usable toiletpaper. If a spent roll is present, transfer it to the garbagecan; then, replenish the toiletpaperhanger with a new toiletpaper from the shelf or cabinet.
- Involved Objects: toiletpaperhanger, toiletpaper, garbagecan, shelf, cabinet
- Required Semantic Action Primitives: navigate_to, open, close, pickup, put_in
#### 3. Bathtub area cleaned and tidied
- Workflow: If the showercurtain is closed, the robot must first open it to check the interior; if there is water in the bathtub, the bathtub is not clean yet and must not be treated as already satisfied, so the robot must clean it first; if there are still objects(cloth, soapbar) in the bathtub that need to be organized, the robot must pickup and transfer them into the cabinet, and finally ensure the showercurtain is left open.
- Involved Objects: showercurtain, bathtub, cloth, soapbar, cabinet
- Required Semantic Action Primitives: navigate_to, open, close, pickup, put_in, clean
#### 4. Sink area cleaned 
- Workflow: If there is water in the sink, the sink is not clean yet and must not be treated as already satisfied, so the robot must clean it; if no soapbottle is detected on the sink, the robot needs to find one from the cabinet or shelf and transfer it onto the sink.
- Involved Objects: sink, soapbottle, cabinet, shelf
- Required Semantic Action Primitives: navigate_to, open, close, pickup, put_in, clean
#### 5. Mirror cleaned and spotless
- Workflow: If fog is observed on the mirror, the robot must clean it to ensure the mirror is spotless.
- Involved Objects: mirror
- Required Semantic Action Primitives: navigate_to, clean
#### 6. Towels neatly arranged
- Workflow: Ensure at least one handtowelholder near the sink is stocked with a handtowel; if none of the holders have one, the robot should find a handtowel from the cabinet or shelf and place it to a handtowelholder.
- Involved Objects: handtowel, handtowelholder, cabinet, shelf
- Required Semantic Action Primitives: navigate_to, pickup, put_in, open, close

### Robot Allocation Rules
- Assign robots to actions based on the action capabilities of each robot.
- If a checklist item cannot be completed by one robot alone, assign multiple robots to cooperate based on their action capabilities.
- Improve housekeeping efficiency by coordinating multiple robots on the same task when needed, and by dispatching idle robots to unfinished checklist items.
- Follow the logical order of actions and do not skip required steps. For example, a robot must open a cabinet before accessing objects inside it, and must navigate near an object before interacting with it.
- For sink and bathtub checklist items, any observed water means the area is not clean yet. Do not mark the item as already satisfied or skip it until the water has been removed by cleaning.
- If the observed room state already satisfies a checklist item, no action is needed for that item. This helps improve bathroom housekeeping efficiency. For example, if a handtowelholder already has a handtowel, that item is already complete.