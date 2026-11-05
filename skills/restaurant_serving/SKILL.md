---
name: restaurant-serving-protocol
description: "Standard menu preparation and delivery protocol. Trigger this skill upon receiving a dining order request. Match the requested dishes against the 'Menu Item SOPs' below, and dynamically allocate robots to execute the corresponding semantic action primitives."
---

## Restaurant Serving Protocol
### Overview
Standard menu preparation and delivery protocol. Trigger this skill upon receiving a dining order request. Match the requested dishes against the 'Menu Item SOPs' below, and dynamically allocate robots to execute the corresponding semantic action primitives.

### Menu Item Standard Operating Procedures
#### 1. Fried Egg Preparation
- Workflow: First, navigate to the countertop to retrieve 1 egg, transfer it to the pan's location, and place it into the pan. Next, process the egg using the cook action. Once cooking is completed, retrieve the prepared egg from the pan and deliver it by placing it onto the diningtable.
- Involved Objects: egg, countertop, pan_1, pan_2, diningtable
- Required Semantic Action Primitives: navigate_to, pickup, cook, put_in

#### 2. Toast Preparation
- Workflow: First, navigate to the countertop to retrieve 1 bread slice, transfer it to the toaster's location, and place it into the toaster. Next, process the bread slice using the cook action. Subsequently, retrieve the toasted bread from the toaster and deliver it by placing it onto the diningtable.
- Involved Objects: bread, breadsliced, countertop, toaster, diningtable
- Required Semantic Action Primitives: navigate_to, pickup, cook, put_in

### Robot Allocation Rules
- Assign robots to actions based on the action capabilities of each robot.
- Maximize meal preparation and delivery efficiency by enabling multi-robot coordination where necessary. Continuously dispatch idle robots to process pending dishes from the order.
- Strictly follow the operational workflow for each dish. Do not skip required steps (e.g., ingredients must be cooked at the designated appliance before being delivered to the diningtable).
- A robot must explicitly perform a navigate_to action to approach a target object or appliance before any interaction (e.g., pickup, cook) can occur.
- For egg orders, use egg as the ingredient name.
- For toast orders, use bread or breadsliced as the ingredient name.
- pan_1, pan_2, and toaster are shared appliances. Each pan can stage/cook up to 2 eggs, and the toaster can stage/cook up to 2 bread slices. An appliance is available for another item while memory shows fewer than 2 staged items; it is full only when memory shows 2/2 staged items.
- For toast, a bread slice must be placed into the toaster before cook is called. If memory marks toaster as occupied with 1/2 staged slice, another bread slice may still be placed into the toaster. If memory marks toaster as full with 2/2 staged slices, do not call put_in(toaster) for another slice; wait for a cook-capable robot to cook and remove staged toast, then deliver it to the diningtable.
- A dish is complete only after the cooked EggCracked or BreadSliced has been delivered to the diningtable. Staging food in pan_1, pan_2, or toaster is not completion.
