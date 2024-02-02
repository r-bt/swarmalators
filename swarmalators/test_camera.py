import cv2

# Get OpenCV build information
build_info = cv2.getBuildInformation()

# Find the list of available cameras
start_index = build_info.find("Video I/O")
end_index = build_info.find("Environment:")
video_io_info = build_info[start_index:end_index].strip()

# Extract camera information
camera_info_start = video_io_info.find("Cameras (not officially supported):")
camera_info_end = video_io_info.find("NULL (nothing has been selected)")
camera_info = video_io_info[camera_info_start:camera_info_end].strip()

# Print the list of available cameras
print(camera_info)
