img_formats = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".dng"]
vid_formats = [".mov", ".avi", ".mp4", ".mpg", ".mpeg", ".m4v", ".wmv", ".mkv"]

IDX_TO_CLASS = {
    0: "background",
    1: "current_lane",
    2: "alternative_lane",
    3: "line",
    4: "dashed_line",
    5: "road_curb",
}

CLASS_TO_IDX = {v: k for k, v in IDX_TO_CLASS.items()}

COLOR_MAPS = {  # BGR Format
    0: (0, 0, 0),  # background - Black
    1: (0, 255, 0),  # current_lane - Green
    2: (255, 0, 0),  # alternative_lane - Blue
    3: (0, 0, 255),  # line - Red
    4: (0, 165, 255),  # dashed_line - Orange
    5: (230, 216, 173),  # road_curb - Light Blue
}
