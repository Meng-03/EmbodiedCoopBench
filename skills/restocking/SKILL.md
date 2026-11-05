---
name: inventory-replenishment
description: "Standard inventory restocking checklist. Trigger this skill upon receiving an inventory audit and replenishment request. Based on the replenishment thresholds of each checklist item and the real-time observed stock levels, dynamically allocate robots to execute the appropriate semantic action primitives."
---

## Inventory Replenishment Checklist
### Overview
Standard inventory restocking checklist. Trigger this skill upon receiving an inventory audit and replenishment request. Based on the replenishment thresholds of each checklist item and the real-time observed stock levels, dynamically allocate robots to execute the appropriate semantic action primitives.

### Restocking Checklist
#### 1. Egg Stock Replenishment
- Workflow: Audit the available quantity of eggs displayed on the countertop. If the stock falls below the threshold of 5 units, the robot must navigate to the inventory reserve: fridge, retrieve the required eggs, and transfer them to the countertop to fulfill the restock quota.
- Involved Objects: egg, countertop, fridge
- Required Semantic Action Primitives: navigate_to, open, pickup, put_in, close

#### 2. Bread Stock Replenishment
- Workflow: Audit the available quantity of bread displayed on the diningtable. If the stock falls below the threshold of 2 units, the robot must navigate to the inventory reserve: cabinet, retrieve the required bread, and transfer it to the diningtable to complete the replenishment.
- Involved Objects: bread, diningtable, cabinet
- Required Semantic Action Primitives: navigate_to, open, pickup, put_in, close

#### 3. Apple Stock Replenishment
- Workflow: Audit the available quantity of apples displayed on the diningtable. If the stock falls below the threshold of 3 units, the robot must navigate to the inventory reserve: fridge, retrieve the apples, and transfer them to the diningtable.
- Involved Objects: apple, diningtable, fridge
- Required Semantic Action Primitives: navigate_to, open, pickup, put_in, close

### Robot Allocation Rules
- Assign robots to actions based on the action capabilities of each robot.
- Assign pickup actions strictly based on the exact tools bound to each robot (pickup_heavy, pickup_medium, or pickup_light). Do not assign a robot to pick up an item if its corresponding pickup capability does not support that item.
- Maximize efficiency by enabling multi-robot coordination where necessary, and by continuously dispatching idle robots to address unfinished checklist items.
- Follow the logical order of actions and do not skip required steps.
- A robot must explicitly perform a navigate_to action to approach a target object in order to reliably observe its real-time stock status before any interaction can occur.
- After concluding all inventory audits and item transfers, the system must perform a final verification to ensure that all inventory reserve units accessed during the operation (i.e., fridge, cabinet) are securely closed.

