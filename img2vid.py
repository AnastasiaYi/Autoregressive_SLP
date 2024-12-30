import cv2
import os

# Path to the folder containing images
image_folder = './Dataset/phoenix2014-release/phoenix-2014-signerindependent-SI5/features/fullFrame-210x260px/train/01April_2010_Thursday_heute_default-0/1'
# Output video file name
video_name = 'output_video.mp4'  # Change to .mp4
# Frame rate for the video
fps = 30

# Get the list of images from the folder
images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
images.sort()  # Sort images by name

# Check if there are any images
if not images:
    print("No images found in the folder.")
    exit()

# Read the first image to get the frame size
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = first_image.shape

# Create a VideoWriter object for MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for MP4 files
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

# Loop through all images and write them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    image_frame = cv2.imread(img_path)
    video.write(image_frame)

# Release the video writer
video.release()

print(f"Video {video_name} created successfully.")