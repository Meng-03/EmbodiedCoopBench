# ============================================================
# 特殊场景配置系统 - 处理特定房间的特殊映射和规则
# ============================================================
class SceneSpecialConfig:
    """
    管理特定场景的特殊配置规则，用于处理诸如对象映射、别名替换等特殊情况。
    支持可扩展的配置，便于添加新的场景特殊规则。
    """
    # 场景特殊规则配置
    # 格式: {
    #     "scene_name": {
    #         "object_mappings": [
    #             {
    #                 "description": "规则描述",
    #                 "source_object_type": "源对象类型（小写）",
    #                 "target_object_type": "目标对象类型（小写）",
    #                 "source_object_id": "源对象 ID（可选），若指定则精确匹配",
    #                 "target_object_id": "目标对象 ID（可选）",
    #                 "apply_to": "适用场景：'put'（放置）/ 'pickup'（拾取）/ 'all'（所有）",
    #                 "condition": 可选的条件函数
    #             }
    #         ]
    #     }
    # }
    SPECIAL_RULES = {
        "FloorPlan402": {
            "description": "浴室 402 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-01.53|+00.93|+04.87",
                    "target_object_id": "CounterTop|-01.02|+00.95|+04.88",
                    "apply_to": "put"  # [put, pickup, all(put和pickup都可以)]
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "countertop",
                    "source_object_id": "Bathtub|-02.90|+00.67|+02.86",
                    "target_object_id":"Bathtub|-02.90|+00.67|+02.86|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        },        
        "FloorPlan403": {
            "description": "浴室 403 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-01.94|+00.95|+03.14",
                    "target_object_id": "CounterTop|-01.36|+01.09|+03.15",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|-02.23|+00.29|+00.72",
                    "target_object_id":"Bathtub|-02.23|+00.29|+00.72|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        },
        "FloorPlan405": {
            "description": "浴室 405 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-03.02|+00.87|+00.21",
                    "target_object_id": "CounterTop|-03.05|+01.04|-00.44",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|-00.60|+00.29|+00.83",
                    "target_object_id":"Bathtub|-00.60|+00.29|+00.83|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        },
        "FloorPlan406": {
            "description": "浴室 406 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|+00.43|+01.04|+02.95",
                    "target_object_id": "CounterTop|+00.49|+01.02|+03.09",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|-01.04|+00.16|+00.56",
                    "target_object_id":"Bathtub|-01.04|+00.16|+00.56|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        },
        "FloorPlan407": {
            "description": "浴室 407 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-00.53|+00.81|-01.60",
                    "target_object_id": "CounterTop|-01.04|+00.88|-01.53",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|+00.88|+00.39|-01.15",
                    "target_object_id":"Bathtub|+00.88|+00.39|-01.15|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        },
        "FloorPlan408": {
            "description": "浴室 408 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-02.68|+00.93|+00.68",
                    "target_object_id": "CounterTop|-02.30|+00.95|+00.69",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|+00.29|+00.29|-00.08",
                    "target_object_id":"Bathtub|+00.29|+00.29|-00.08|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        },        
        "FloorPlan413": {
            "description": "浴室 413 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-01.83|+00.69|+03.47",
                    "target_object_id":"CounterTop|-01.80|+00.75|+03.12",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|-01.41|00.00|+00.56",
                    "target_object_id":"Bathtub|-01.41|00.00|+00.56|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        }, 
        "FloorPlan414": {
            "description": "浴室 414 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-02.72|+00.86|+02.23",
                    "target_object_id":"CounterTop|-02.69|+00.97|+02.26",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|-00.44|+00.18|+01.58",
                    "target_object_id":"Bathtub|-00.44|+00.18|+01.58|BathtubBasin",
                    "apply_to": "put"  
                },                
            ]
        },        
        "FloorPlan422": {
            "description": "浴室 422 的特殊规则",
            "object_mappings": [
                {
                    "description": "Sink 操作映射到 CounterTop - 房间中有两个 Sink，put/pickup 实际映射到 CounterTop",
                    "source_object_type": "sink",
                    "target_object_type": "countertop",
                    "source_object_id": "Sink|-01.93|+00.77|+00.33",
                    "target_object_id":"CounterTop|-01.92|+00.00|+00.29",
                    "apply_to": "put"  
                },
                {
                    "description": "Bathtub 操作映射到 BathtubBasin",
                    "source_object_type": "bathtub",
                    "target_object_type": "bathtubbasin",
                    "source_object_id": "Bathtub|-03.45|+00.18|+01.03",
                    "target_object_id":"Bathtub|-03.45|+00.18|+01.03|BathtubBasin",
                    "apply_to": "put"  
                }
            ]
        }, 
        # 未来可以添加更多场景的规则
        # "FloorPlan403": { ... }
        
    }
    
    @classmethod
    def get_scene_rules(cls, scene_name):
        """获取指定场景的所有规则"""
        return cls.SPECIAL_RULES.get(scene_name, {})
    
    @classmethod
    def has_rules(cls, scene_name):
        """检查场景是否有特殊规则"""
        return scene_name in cls.SPECIAL_RULES
    
    @classmethod
    def apply_object_mapping(cls, scene_name, source_obj_type, target_object_id, 
                             operation_type="all", source_object_id=None):
        """
        应用对象映射规则，将源对象类型映射到目标对象 ID
        
        Args:
            scene_name: 场景名称
            source_obj_type: 源对象类型（小写）
            target_object_id: 目标对象 ID（当前正在处理的）
            operation_type: 操作类型 ('put', 'pickup', 'all')
            source_object_id: 源对象 ID（可选的精确匹配）
        
        Returns:
            (是否应应用映射, 目标对象 ID) 的元组
            如果应用映射则返回 (True, 新的目标对象 ID)
            否则返回 (False, 原目标对象 ID)
        """
        # print("mapping!")
        if not cls.has_rules(scene_name):
            return False, target_object_id
        
        rules = cls.get_scene_rules(scene_name)
        mappings = rules.get("object_mappings", [])
        
        for mapping in mappings:
            # 检查源对象类型是否匹配
            if mapping.get("source_object_type") != source_obj_type: continue
            
            # 检查操作类型是否匹配
            apply_to = mapping.get("apply_to", "all")
            if apply_to != "all" and apply_to != operation_type: continue
            
            # 检查源对象 ID 是否需要精确匹配
            if "source_object_id" in mapping:
                if source_object_id and source_object_id != mapping["source_object_id"]:
                    continue
            
            # 应用映射，返回目标对象 ID
            mapped_obj_id = mapping.get("target_object_id")
            if mapped_obj_id:
                return True, mapped_obj_id
        
        return False, target_object_id

    @classmethod
    def get_valid_receptacle_ids(cls, scene_name, receptacle_id):
        """
        获取一个容器 ID 在指定场景下的所有有效等价 ID 集合。
        用于 eval 时扩展 parentReceptacleId 的判定范围。
        
        例如: FloorPlan402 中 goal 要求物品放在 Sink|-01.53|+00.93|+04.87，
        但实际 put 被映射到了 CounterTop|-01.02|+00.95|+04.88，
        两者都应视为合法的父容器。
        
        Args:
            scene_name: 场景名称
            receptacle_id: goal_verification 中的目标容器 ID
        
        Returns:
            set: 包含原始 ID 及所有映射等价 ID 的集合
        """
        valid_ids = {receptacle_id}
        
        if not cls.has_rules(scene_name):
            return valid_ids
        
        rules = cls.get_scene_rules(scene_name)
        mappings = rules.get("object_mappings", [])
        
        for mapping in mappings:
            source_id = mapping.get("source_object_id")
            target_id = mapping.get("target_object_id")
            apply_to = mapping.get("apply_to", "all")
            
            # 只对 put 相关的映射做扩展（parentReceptacleId 本质是 put 的目标）
            if apply_to not in ["put", "all"]:
                continue
            
            # 如果 goal 中的容器是映射的 source，则 target 也合法
            if source_id == receptacle_id and target_id:
                valid_ids.add(target_id)
            # 如果 goal 中的容器是映射的 target，则 source 也合法
            if target_id == receptacle_id and source_id:
                valid_ids.add(source_id)
        
        return valid_ids

    @classmethod
    def print_all_rules(cls):
        """打印所有场景规则"""
        for scene_name, config in cls.SPECIAL_RULES.items():
            print(f"\n📋 场景 {scene_name}:")
            print(f"   描述: {config.get('description', 'N/A')}")
            mappings = config.get("object_mappings", [])
            for idx, mapping in enumerate(mappings, 1):
                print(f"   规则 {idx}: {mapping.get('description', 'N/A')}")
                print(f"      来源: {mapping.get('source_object_type')} -> 目标: {mapping.get('target_object_type')}")
