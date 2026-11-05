
AGENT_STEP_SIZE = 0.25 #0.25
HORIZON_GRANULARITY = 15 #视野 粒度
CAMERA_HEIGHT_OFFSET = 0.75

WIDTH = 800 #2000
HEIGHT = 450 #1000

THOR_CONTROLLER_RESET_RETRIES = 3
THOR_CONTROLLER_RESTART_BACKOFF_SEC = 15.0
THOR_CONTROLLER_RESET_TIMEOUT_SEC = 20.0
THOR_CONTROLLER_RESET_TIMEOUT_RETRIES = 1
THOR_CONTROLLER_TIMEOUT_RESTART_BACKOFF_SEC = 0.0

# force a different horizon view for objects of (type, location). If the location is None, force this horizon for all objects of that type.
FORCED_HORIZON_OBJS = {
    ("FloorLamp", None): 0,
    ("Fridge", 18): 30,
    ("Toilet", None): 15,
}

OPENABLE_OBJECT_TYPE=[ #openable recepable
    "Box",
    "Cabinet",
    "Drawer",
    "Fridge",
    "Microwave",
    "Safe",
    "Toilet"
]