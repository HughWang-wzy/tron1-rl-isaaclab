"""Terrain configurations for navigation tasks."""

from isaaclab.terrains import (
    HfDiscreteObstaclesTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,
    MeshRepeatedBoxesTerrainCfg,
    TerrainGeneratorCfg,
)


#############################
# Navigation Obstacle Terrain
#############################

NAV_OBSTACLES_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.4,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.05, 0.15),
            obstacle_width_range=(0.1, 0.5),
            num_obstacles=40,
            platform_width=2.0,
        ),
        "random_boxes": MeshRandomGridTerrainCfg(
            proportion=0.3,
            grid_width=0.45,
            grid_height_range=(0.02, 0.10),
            platform_width=2.0,
        ),
    },
    curriculum=False,
)


NAV_OBSTACLES_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
        "obstacles": HfDiscreteObstaclesTerrainCfg(
            proportion=0.4,
            obstacle_height_mode="fixed",
            obstacle_height_range=(0.05, 0.15),
            obstacle_width_range=(0.1, 0.5),
            num_obstacles=40,
            platform_width=2.0,
        ),
        "random_boxes": MeshRandomGridTerrainCfg(
            proportion=0.3,
            grid_width=0.45,
            grid_height_range=(0.02, 0.10),
            platform_width=2.0,
        ),
    },
    curriculum=False,
)
