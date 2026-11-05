
from PIL import Image,ImageDraw
import numpy as np
import json
import os
import cv2
import env.ai2thor.constants as constants
import matplotlib.pyplot as plt
from datetime import datetime

import logging
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO
    format="%(asctime)s - %(levelname)s - %(message)s",  # 日志输出格式
)


def _as_grid_point_array(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.int32)
    return arr.reshape((-1, 2))


# 将 metadata 中的 image 保存
def save_one_frame(frame,path):
    """
    save one frame to path
    """
    parent_dir=os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir,exist_ok=True)
    
    image=Image.fromarray(frame)
    image.save(path)
    print(f"[Success!] image save to {path}")
    
    
def image_chunking(image_path,save_path,grid_size_x=200,grid_size_y=300):
    image=Image.open(image_path)
    width,height=image.size
    print("image width",width,"image height",height)
    draw=ImageDraw.Draw(image)
    
    for x in range(0,width,grid_size_x):
        draw.line((x,0,x,height),fill="white",width=2)
    for y in range(0,height,grid_size_y):
        draw.line((0,y,width,y),fill="white",width=2) 
    image.save(save_path)
    print(f"[Success!] image save to {save_path}")

def draw_point_on_image(image_path, save_path, x, y, color='white', radius=5):
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    # 画点
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
    # 保存新的图像
    image.save(save_path)
    print(f"[Success!] Image saved to {save_path}")

def save_metadata(metadata,save_path):
    """
    保存metadata
    """
    with open(save_path,"w") as f:
        json.dump(metadata,f,indent=4)
    print(f"[Success!] Metadata saved to {save_path}")
    
    

def make_topview_video(folder, video_name="topview.mp4", fps=5):
    # 1. 收集所有 *_topview.jpg
    frames = [f for f in os.listdir(folder) if f.endswith("_topview.jpg")]
    if not frames:
        print("No topview frames found.")
        return
    # 2. 按前缀数字排序，比如 0_topview.jpg, 1_topview.jpg, ...
    frames.sort(key=lambda x: int(x.split("_", 1)[0]))
    # 3. 读第一张图，确定分辨率
    first_path = os.path.join(folder, frames[0])
    img = cv2.imread(first_path)
    h, w = img.shape[:2]
    # 4. 创建 VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(os.path.join(folder, video_name), fourcc, fps, (w, h))
    # 5. 依次写入所有帧
    for name in frames:
        path = os.path.join(folder, name)
        frame = cv2.imread(path)
        if frame is not None:
            out.write(frame)
    out.release()
    print("Saved video to:", os.path.join(folder, video_name))
    
    
    
    
# gotolocation相关：与物体交互的位姿
# If known_point is given, calculate the nearest point to that known_point, else calculate the nearest point
# to the center of the object given in the box.
def get_obj_coords(
    box,
    points,
    camera_height,
    known_point=None,
    object_type=None,
    current_scene=None
):
    # obj_name = (cls, tuple(box.flatten().tolist()))
    center = (box[:, 0] + box[:, 1]) / 2  # Boxes are in xzy order
    point_dists = center[:2][np.newaxis, :] - points
    point_dists_mag = np.sum(np.abs(point_dists), axis=1)
    # Make lint happy
    best_dist = None
    best_point = None
    if known_point is not None:
        best_point = (known_point[0], known_point[1])
        best_dist = center[:2] - known_point
    # If the best point is not given, approximate it.
    if best_point is None:
        best_loc = np.argmin(point_dists_mag)
        best_dist = point_dists[best_loc, :]
        best_point = (points[best_loc, 0], points[best_loc, 1])
    best_point = np.array(best_point)
    # Get rotation and camera angle.
    dist_to_obj = (
        np.sqrt(
            np.sum(
                np.square(
                    np.array(
                        [
                            best_point[0],
                            best_point[1],
                            camera_height / constants.AGENT_STEP_SIZE,
                        ]
                    )
                    - center
                )
            )
        )
        * constants.AGENT_STEP_SIZE
    )
    if abs(best_dist)[0] > abs(best_dist[1]):
        # X value
        if best_dist[0] > 0:
            # object is to the east of ground point
            rotation = 1
        else:
            # object is to the west of ground point
            rotation = 3
    else:
        if best_dist[1] > 0:
            # object is to the north of ground point
            rotation = 0
        else:
            # object is to the south of ground point
            rotation = 2
    if dist_to_obj < 0.5:  # pull back if too close
        new_best_point = best_point.copy()
        if rotation == 0:
            new_best_point[1] -= 1
        elif rotation == 1:
            new_best_point[0] -= 1
        elif rotation == 2:
            new_best_point[1] += 1
        else:
            new_best_point[0] += 1
        new_point_dists = (new_best_point[np.newaxis, :] - points).astype(np.int32)
        new_point_dists_mag = np.sum(np.abs(new_point_dists), axis=1)
        new_best_loc = np.argmin(new_point_dists_mag)
        if new_point_dists_mag[new_best_loc] == 0:
            # new best point exists, use it
            best_loc = new_best_loc
            best_dist = point_dists[best_loc, :]
            best_point = np.array([points[best_loc, 0], points[best_loc, 1]])
    horizontal_dist_to_obj = np.max(np.abs(best_dist)) * constants.AGENT_STEP_SIZE
    # Don't use center y, use bottom
    obj_height = camera_height - box[2, 0] * constants.AGENT_STEP_SIZE
    camera_angle = int(
        np.clip(
            np.round(
                np.arctan2(obj_height, horizontal_dist_to_obj)
                * (180 / np.pi / constants.HORIZON_GRANULARITY)
            )
            * constants.HORIZON_GRANULARITY,
            -30,
            60,
        )
    )

    # Hard overwrites for camera_angle adjustments based on object type and scene.
    if object_type is not None and current_scene is not None:
        if (object_type, current_scene) in constants.FORCED_HORIZON_OBJS:
            camera_angle = constants.FORCED_HORIZON_OBJS[
                (object_type, current_scene)
            ]
        elif (object_type, None) in constants.FORCED_HORIZON_OBJS:
            camera_angle = constants.FORCED_HORIZON_OBJS[(object_type, None)]

    coord = (
        int(best_point[0]),
        int(best_point[1]),
        int(rotation),
        int(camera_angle),
    )
    return coord


def get_obj_coords_top_k(
    box,
    points,
    camera_height,
    known_point=None,
    object_type=None,
    current_scene=None,
    top_k=3
):
    # obj_name = (cls, tuple(box.flatten().tolist()))
    center = (box[:, 0] + box[:, 1]) / 2  # Boxes are in xzy order
    point_dists = center[:2][np.newaxis, :] - points
    point_dists_mag = np.sum(np.abs(point_dists), axis=1)
    
    # 获取候选点
    top_indices = np.argsort(point_dists_mag)[:top_k]
    candidates=[]
    
    for best_loc in top_indices:
        # Make lint happy
        best_dist = None
        best_point = None
        if known_point is not None: #todo.未适配 top k 个 known point
            best_point = (known_point[0], known_point[1])
            best_dist = center[:2] - known_point
        # If the best point is not given, approximate it.
        if best_point is None:
            # best_loc = np.argmin(point_dists_mag)
            best_dist = point_dists[best_loc, :]
            best_point = (points[best_loc, 0], points[best_loc, 1])
        best_point = np.array(best_point)
        # Get rotation and camera angle.
        dist_to_obj = (
            np.sqrt(
                np.sum(
                    np.square(
                        np.array(
                            [
                                best_point[0],
                                best_point[1],
                                camera_height / constants.AGENT_STEP_SIZE,
                            ]
                        )
                        - center
                    )
                )
            )
            * constants.AGENT_STEP_SIZE
        )
        if abs(best_dist)[0] > abs(best_dist[1]):
            # X value
            if best_dist[0] > 0:
                # object is to the east of ground point
                rotation = 1
            else:
                # object is to the west of ground point
                rotation = 3
        else:
            if best_dist[1] > 0:
                # object is to the north of ground point
                rotation = 0
            else:
                # object is to the south of ground point
                rotation = 2
        if dist_to_obj < 0.5:  # pull back if too close
            new_best_point = best_point.copy()
            if rotation == 0:
                new_best_point[1] -= 1
            elif rotation == 1:
                new_best_point[0] -= 1
            elif rotation == 2:
                new_best_point[1] += 1
            else:
                new_best_point[0] += 1
            new_point_dists = (new_best_point[np.newaxis, :] - points).astype(np.int32)
            new_point_dists_mag = np.sum(np.abs(new_point_dists), axis=1)
            new_best_loc = np.argmin(new_point_dists_mag)
            if new_point_dists_mag[new_best_loc] == 0:
                # new best point exists, use it
                best_loc = new_best_loc
                best_dist = point_dists[best_loc, :]
                best_point = np.array([points[best_loc, 0], points[best_loc, 1]])
        horizontal_dist_to_obj = np.max(np.abs(best_dist)) * constants.AGENT_STEP_SIZE
        # Don't use center y, use bottom
        obj_height = camera_height - box[2, 0] * constants.AGENT_STEP_SIZE
        camera_angle = int(
            np.clip(
                np.round(
                    np.arctan2(obj_height, horizontal_dist_to_obj)
                    * (180 / np.pi / constants.HORIZON_GRANULARITY)
                )
                * constants.HORIZON_GRANULARITY,
                -30,
                60,
            )
        )

        # Hard overwrites for camera_angle adjustments based on object type and scene.
        if object_type is not None and current_scene is not None:
            if (object_type, current_scene) in constants.FORCED_HORIZON_OBJS:
                camera_angle = constants.FORCED_HORIZON_OBJS[
                    (object_type, current_scene)
                ]
            elif (object_type, None) in constants.FORCED_HORIZON_OBJS:
                camera_angle = constants.FORCED_HORIZON_OBJS[(object_type, None)]

        coord = (
            int(best_point[0]),
            int(best_point[1]),
            int(rotation),
            int(camera_angle),
        )
        candidates.append(coord)
    
    return candidates


def compute_best_nav_points_for_all_objects(
    controller,
    known_object_points=None,
    forced_horizon_objs=None,
    object_filter=None,
):
    """
    参考 ai2thor / ALFRED 的实现：
    为当前场景中的所有物体计算“最佳导航点 + 朝向 + 相机角度”。

    :param controller: ai2thor.Controller 实例
    :param constants.AGENT_STEP_SIZE: 栅格大小（米），例如 0.25
    :param camera_height: 相机高度（米），例如 1.575
    :param horizon_granularity: 相机角度离散粒度（度），例如 5
    :param known_object_points: dict(objectId -> (x,z) 世界坐标)，可选的预计算最佳点
    :param forced_horizon_objs: dict((object_type, scene) -> horizon) 的强制表，可选
    :param object_filter: 一个 objId列表，用来过滤只关心的物体，默认不过滤
    :return: dict: objectId -> (x_idx, z_idx, rotation, camera_angle)
    """
    # TODO. 多个点选择，不要相同一个点（不过CBS中也可以先后到达）
    agent_height = controller.last_event.metadata['agent']['position']['y']
    camera_height = agent_height + constants.CAMERA_HEIGHT_OFFSET
    
    if known_object_points is None:
        known_object_points = {}
    if forced_horizon_objs is None:
        forced_horizon_objs = {}

    # 1. 获取可达点（世界坐标）
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]  # list of dicts: {'x':..., 'y':..., 'z':...}

    # 转成栅格坐标 (x_idx, z_idx)
    points = []
    for pos in reachable_positions:
        x_idx = int(round(pos["x"] / constants.AGENT_STEP_SIZE))
        z_idx = int(round(pos["z"] / constants.AGENT_STEP_SIZE))
        points.append([x_idx, z_idx])
    points = _as_grid_point_array(points)  # (N, 2)

    # 筛选掉当前agent的位置
    agent_positions = []
    for e in controller.last_event.events:
        agent_pos = [int(round(e.metadata['agent']['position']['x'] / constants.AGENT_STEP_SIZE)),
                     int(round(e.metadata['agent']['position']['z'] / constants.AGENT_STEP_SIZE))]
        agent_positions.append(agent_pos)
    
    # 过滤 points，移除掉所有 agent 占据的位置。
    # 若过滤后为空，则回退到原始 reachable points，避免在狭小区域/单点可达时
    # 将当前站位也移除掉，从而导致后续候选导航点计算直接退化为异常。
    filtered_points = _as_grid_point_array([
        p for p in points
        if not any(p[0] == ap[0] and p[1] == ap[1] for ap in agent_positions)
    ])
    if filtered_points.size > 0:
        points = filtered_points

    # 2. 拿当前场景中所有物体
    event = controller.step(action="Pass")  # 或者直接用 controller.last_event
    objects = event.metadata["objects"]

    scene_name = event.metadata["sceneName"]
    
    current_scene = scene_name

    objid_to_coord = {}

    for obj in objects:
        obj_type = obj["objectType"]
        obj_id = obj["objectId"]

        if object_filter is not None and obj_id not in object_filter:
            continue  # 让调用者决定“哪些物体需要导航点”

        # 3. 构建简化版 box（用 position 当成 min/max）
        pos = obj["position"]
        # 注意这里是 x,z,y 的顺序
        box = np.array(
            [
                [pos["x"], pos["x"]],
                [pos["z"], pos["z"]],
                [pos["y"], pos["y"]],
            ],
            dtype=np.float32,
        ) / constants.AGENT_STEP_SIZE

        # 4. 是否有预先给的最佳点（世界坐标），有则转成栅格坐标
        known_point = None
        if obj_id in known_object_points:
            kp = known_object_points[obj_id]
            known_point = np.array(
                [kp[0] / constants.AGENT_STEP_SIZE, kp[1] / constants.AGENT_STEP_SIZE], dtype=np.float32
            )

        coord = get_obj_coords_top_k(
            box=box,
            points=points,
            camera_height=camera_height,
            known_point=known_point,
            object_type=obj_type,
            current_scene=current_scene,
        )

        objid_to_coord[obj_id] = coord

    return objid_to_coord
   
    
    
# gotoLocation相关：输入到 cbs 算法前，grid plot的转换
def plot_cbs_grid(size, reachable_shifted, obstacles):
    """
    Plot a CBS-style grid:
    - blue dots = reachable
    - red dots = obstacles
    """
    fig, ax = plt.subplots()

    # Convert sets to lists for plotting
    reachable_x = [p[0] for p in reachable_shifted]
    reachable_z = [p[1] for p in reachable_shifted]

    obstacle_x = [p[0] for p in obstacles]
    obstacle_z = [p[1] for p in obstacles]

    # ---- Plot reachable points (blue) ----
    ax.scatter(reachable_x, reachable_z, s=40)

    # ---- Plot obstacles (red) ----
    ax.scatter(obstacle_x, obstacle_z, s=40)

    # ---- Grid style ----
    ax.set_xticks(np.arange(0, size, 1))
    ax.set_yticks(np.arange(0, size, 1))
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_aspect('equal', 'box')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Axis labels
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_title(f"Grid Plot (size={size})")

    plt.show()

def grid_plot(controller,goto_robot_ids,scene="FloorPlan_idx"):
    # timestamp = datetime.now().strftime("%Y%m%d")

    unreachable_robot_pos = []
    for i,e in enumerate(controller.last_event.events):
        if i not in goto_robot_ids:
            # todo. something wrong with unreachable_robot_pos
            unreachable_robot_pos.append(e.metadata['agent']['position'])
        # print(f"{e.metadata['agentId']}",agentpos)
        
    event = controller.step(action="GetReachablePositions")
    positions = event.metadata["actionReturn"]
    # 过滤掉在 unreachable_robot_pos 中的位置
    filtered_positions = [
        p for p in positions
        if not any(p['x'] == unp['x'] and p['z'] == unp['z'] for unp in unreachable_robot_pos)
    ]
    x = [p['x'] for p in filtered_positions]
    z = [p['z'] for p in filtered_positions]    
    # for debug记录
    # fig, ax = plt.subplots()
    # ax.scatter(x,z)
    # print('getreachablepoints')
    
    # ---- Compute grid dimensions and unreachable world coordinates ----
    # Round world coords to unique grid samples
    x_rounded = sorted(set(round(px/constants.AGENT_STEP_SIZE) for px in x))
    z_rounded = sorted(set(round(pz/constants.AGENT_STEP_SIZE) for pz in z))
    xmin, xmax = min(x_rounded), max(x_rounded)
    zmin, zmax = min(z_rounded), max(z_rounded)
    # Represent reachable world cells as rounded coordinate tuples
    reachable_cells = set(
        (round(px/constants.AGENT_STEP_SIZE), round(pz/constants.AGENT_STEP_SIZE)) for px, pz in zip(x, z)
    )
    # ---- 将坐标整体偏移到从 0 开始 ----
    # x' = x - xmin, z' = z - zmin
    reachable_shifted = set(
        (xx - xmin, zz - zmin) for (xx, zz) in reachable_cells
    )
    
    width = xmax - xmin + 1   # 原始 X 方向网格数量
    height = zmax - zmin + 1  # 原始 Z 方向网格数量
    # ---- 让 dimensions 变成方形 ----
    size = max(width, height)   # 正方形边长

    obstacles = set()   
    # 先填充原始范围内的“未被 reachable 占据”的网格为障碍
    for ix in range(width):        # 0 .. width-1
        for iz in range(height):   # 0 .. height-1
            if (ix, iz) not in reachable_shifted:
                obstacles.add((ix, iz))

    # 如果 X 方向较短：在右侧补充至 size，全部作为障碍
    if width < size:
        for ix in range(width, size):   # 右侧补充列
            for iz in range(size):      # 整列都作为障碍（包括原高度和补下面的）
                obstacles.add((ix, iz))

    # 如果 Z 方向较短：在下侧补充至 size，全部作为障碍
    if height < size:
        for ix in range(width):              # 只在已有宽度范围内
            for iz in range(height, size):   # 底部补充行
                obstacles.add((ix, iz))

    # 至此，grid 范围是 [0, size-1] x [0, size-1]，dimensions 为方形

    # print("map for CBS-like input:")
    # # 如果你现在希望 dimensions 是“大小”而不是“索引范围”，用下面一行：
    # print(f"  dimensions: [{size}, {size}]")
    # print("  obstacles:")
    
    cbs_dim = [size,size]
    
    cbs_obs = []
    for ox, oz in sorted(obstacles):
        cbs_obs.append((ox,oz))
    
    # 查看 cbs 图片是否正确
    plot_cbs_grid(size,reachable_shifted,obstacles)
    
    return cbs_dim,cbs_obs,xmin,zmin
    
    
    x1,z1 =[],[]
    for e in controller.last_event.events: 
        agentpos = e.metadata['agent']['position']
        # print(agentpos)
        x1.append(agentpos['x'])
        z1.append(agentpos['z'])
    ax.scatter(x1,z1,alpha=0.5)
    
    ax.set_xlabel("X (world coordinate)")
    ax.set_ylabel("Z (world coordinate)")
    
    xmin, xmax = min(x), max(x)
    zmin, zmax = min(z), max(z)
    ax.set_xticks(np.arange(xmin, xmax + constants.AGENT_STEP_SIZE, constants.AGENT_STEP_SIZE))
    ax.set_yticks(np.arange(zmin, zmax + constants.AGENT_STEP_SIZE, constants.AGENT_STEP_SIZE))
    out_dir = f"log/{timestamp}/grid_plot/"
    out_path = os.path.join(out_dir, f"grid_plot_{scene}.png")
    os.makedirs(out_dir, exist_ok=True)
    # 保存当前 Figure
    fig.savefig(out_path, dpi=300, bbox_inches='tight')  # 可改 dpi/格式
    # plt.show()
    plt.close(fig) 
    
    # controller.stop()
    
    # # 1) X 轴正方向（从(0,0)向右）
    # ax.annotate(
    #     '', xy=(max(x), 0), xytext=(0, 0),
    #     arrowprops=dict(arrowstyle="->", linewidth=1, color='red')
    # )
    # ax.text(max(x), 0, " +X", fontsize=12, verticalalignment='bottom')

    # # 2) Z 轴正方向（从(0,0)向上）
    # ax.annotate(
    #     '', xy=(0, max(z)), xytext=(0, 0),
    #     arrowprops=dict(arrowstyle="->", linewidth=1, color='red')
    # )
    # ax.text(0, max(z), " +Z", fontsize=12, horizontalalignment='left')

# 计算到各个标准方向的旋转距离（考虑循环）
def get_dist(a1, a2):
    # print(a1,a2)
    return (a2 - a1 + 180) % 360 - 180

def path_to_actions(robot_id,path, initial_rotation,initial_pitch,target_rotation, target_pitch):
    """
    TODO. cbs 输出的 path 中还考虑了 t，转换时暂时没考虑 t
    {0: [{'t': 0, 'x': 0, 'y': 7}, {'t': 1, 'x': 0, 'y': 8}], 1:[]}
    """
    ll_actions = []
    # --1. 修正初始角度 ---
    # 1.1 轴角度
    # 找到最近的标准方向 (0, 90, 180, 270)
    standard_angles = [0, 90, 180, 270]

    closest_rot = min(standard_angles, key=lambda x: abs(get_dist(initial_rotation, x)))
    correction_dist = get_dist(initial_rotation, closest_rot)
    if correction_dist != 0:
        ll_actions.append({
            "action":"RotateRight",
            "degrees":correction_dist,
            "agentId":int(robot_id)
        })
        
    current_rot = closest_rot # 现在机器人已经对齐标准方向
    
    # 1.2 俯仰角度 [-60,-30,0,30]
    # 向下看为正，向上为负
    p = max(-60, min(30, initial_pitch))
    p = round(p/30)*30
    steps = (0-p)//30
    if steps > 0:
        ll_actions.append({"action": "LookUp", "degrees": 30* int(steps),"agentId":int(robot_id)})
    if steps < 0:
        ll_actions.append({"action": "LookDown", "degrees": 30* int(-steps),"agentId":int(robot_id)})
    
    # --2. 正常的路径转换 ---
    for i in range(len(path) - 1):
        curr, nxt = path[i], path[i+1]
        dx, dy = nxt['x'] - curr['x'], nxt['y'] - curr['y']
        
        if dy > 0: target_rot = 0
        elif dx > 0: target_rot = 90
        elif dy < 0: target_rot = 180
        elif dx < 0: target_rot = 270
        else: continue
            
        diff = (target_rot - current_rot + 360) % 360
        if diff == 90:
            ll_actions.append({
                "action":"RotateRight",
                "degrees":90,
                "agentId":int(robot_id)
            })
        elif diff == 180:
            ll_actions.append({
                "action":"RotateRight",
                "degrees":90,
                "agentId":int(robot_id)
            })
            ll_actions.append({
                "action":"RotateRight",
                "degrees":90,
                "agentId":int(robot_id)
            })
        elif diff == 270:
            ll_actions.append({
                "action":"RotateLeft",
                "degrees":90,
                "agentId":int(robot_id)
            })
        
        ll_actions.append({
            "action":"MoveAhead",
            "moveMagnitude":constants.AGENT_STEP_SIZE,
            "agentId":int(robot_id)
        })
        current_rot = target_rot
        
    # --3. 最终角度对齐 ---
    if target_rotation == 0:
        target_rotation = 0
    elif target_rotation == 1:
        target_rotation = 90
    elif target_rotation == 2:
        target_rotation = 180
    elif target_rotation == 3:
        target_rotation = 270    
    ll_actions.append({ 
        "action":"Teleport", #未设置的参数 remains the same
        "rotation":dict(x=0,y=target_rotation,z=0),
        "horizon":target_pitch,
        "agentId":int(robot_id)
    })    
    
    return ll_actions