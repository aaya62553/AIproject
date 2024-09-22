from utils import (read_video, save_video, get_video_fps,measure_distance,convert_pixels_to_meters)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2

def main():
  #Read video
  input_video_path="input_videos/input_video.mp4"
  video_frames=read_video(input_video_path)
  fps=get_video_fps(input_video_path)
  #Detect players
  player_tracker=PlayerTracker(model_path="yolov8x")
  ball_tracker=BallTracker(model_path="models/last.pt")


  player_detections=player_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/player_detections.pkl")
  ball_detections=ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
  ball_detections=ball_tracker.interpolate_ball_positrions(ball_detections)
  #Detect court lines
  court_line_detector=CourtLineDetector("models/keypoints_model.pth")
  court_keypoints=court_line_detector.predict(video_frames[100])

  #choose players
  player_detections=player_tracker.choose_and_filter_players(court_keypoints,player_detections)

  #Initialize mini court
  mini_court=MiniCourt(video_frames[0])

  #Ball shot frames
  ball_shot_frames=ball_tracker.get_ball_shot_frames(ball_detections)

  player_mini_court_detections,ball_mini_court_detections=mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                      ball_detections,
                                                                                                                      court_keypoints)

  for ball_shot_ind in range(len(ball_shot_frames)-1):
    start_frame=ball_shot_frames[ball_shot_ind]
    end_frame=ball_shot_frames[ball_shot_ind+1]
    ball_shot_time_seconds=(end_frame-start_frame)/fps

    #get distance covered by the ball
    ball_distance_covered_pixels=measure_distance(ball_mini_court_detections[start_frame][1],ball_mini_court_detections[end_frame][1])
    ball_distance_covered_meters=convert_pixels_to_meters(ball_distance_covered_pixels)

  #Draw bounding boxes
  output_video_frames=player_tracker.draw_bboxes(video_frames,player_detections)
  output_video_frames=ball_tracker.draw_bboxes(output_video_frames,ball_detections)
  output_video_frames=court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)


  #Draw mini court
  output_video_frames=mini_court.draw_mini_court(output_video_frames)

  output_video_frames=mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
  output_video_frames=mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections,color=(0,255,0))

  #Draw frame number
  for i,frame in enumerate(output_video_frames):
    cv2.putText(frame, f"Frame : {i}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  #Save video
  save_video(output_video_frames,"output_videos/output_video.avi",fps=fps)

if __name__ == "__main__":
  main()