from utils import (read_video, save_video, get_video_fps,measure_distance,convert_pixel_distance_to_meters,draw_player_stats)
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
import constants
from copy import deepcopy
import pandas as pd 
import os

def main(input_video_path="input_videos/input_video.mp4"):
  #Read video
  video_name=input_video_path.split("/")[-1].split(".")[0]
  video_frames=read_video(input_video_path)
  fps=get_video_fps(input_video_path)
  #Detect players
  player_tracker=PlayerTracker(model_path="yolov8x")
  ball_tracker=BallTracker(model_path="models/last.pt")

  if f"player_detections_{video_name}.pkl" in os.listdir("tracker_stubs") and f"ball_detections_{video_name}.pkl" in os.listdir("tracker_stubs"):
    read_from_stub=True
  else:
    read_from_stub=False

  player_detections=player_tracker.detect_frames(video_frames,
                                                 read_from_stub=read_from_stub,
                                                 stub_path=f"tracker_stubs/player_detections_{video_name}.pkl")
  ball_detections=ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=read_from_stub,
                                                 stub_path=f"tracker_stubs/ball_detections_{video_name}.pkl")
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
  player_stats_data=[{"frame_number":0,
                      "player_1_number_of_shots":0,
                      "player_1_total_shot_speed":0,
                      "player_1_last_shot_speed":0,
                      "player_1_total_player_speed":0,
                      "player_1_last_player_speed":0,
                      
                      "player_2_number_of_shots":0,
                      "player_2_total_shot_speed":0,
                      "player_2_last_shot_speed":0,
                      "player_2_total_player_speed":0,
                      "player_2_last_player_speed":0}]

  for ball_shot_ind in range(len(ball_shot_frames)):
    start_frame=ball_shot_frames[ball_shot_ind]
    end_frame=ball_shot_frames[ball_shot_ind+1] if ball_shot_ind+1<len(ball_shot_frames) else len(video_frames)-1
    ball_shot_time_seconds=(end_frame-start_frame)/fps

    #get distance covered by the ball
    ball_distance_covered_pixels=measure_distance(ball_mini_court_detections[start_frame][1],ball_mini_court_detections[end_frame][1])
    ball_distance_covered_meters=convert_pixel_distance_to_meters(ball_distance_covered_pixels,
                                                                  constants.DOUBLE_LINE_WIDTH,
                                                                  mini_court.get_width_mini_court())
    # print(f"Ball shot distance covered : {ball_distance_covered_meters} meters")
    #get speed of the ball
    speed_ball_shot=3.6*ball_distance_covered_meters/ball_shot_time_seconds 
    # print(f"Ball shot speed : {speed_ball_shot} km/h")

    #player who shot the ball
    player_positions=player_mini_court_detections[start_frame]
    player_shot_ball=min(player_positions.keys(),key=lambda player_id:measure_distance(player_positions[player_id],
                                                                                       ball_mini_court_detections[start_frame][1]))
    #opponent player speed
    opponent_player_id=1 if player_shot_ball==2 else 2
    distance_covered_opponent_pixels=measure_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                      player_mini_court_detections[end_frame][opponent_player_id])
    speed_opponent=3.6*convert_pixel_distance_to_meters(distance_covered_opponent_pixels, 
                                                        constants.DOUBLE_LINE_WIDTH,
                                                        mini_court.get_width_mini_court())/ball_shot_time_seconds
    current_player_stats=deepcopy(player_stats_data[-1])
    current_player_stats["frame_number"]=start_frame
    current_player_stats[f"player_{player_shot_ball}_number_of_shots"]+=1
    current_player_stats[f"player_{player_shot_ball}_total_shot_speed"]+=speed_ball_shot
    current_player_stats[f"player_{player_shot_ball}_last_shot_speed"]=speed_ball_shot
    current_player_stats[f"player_{opponent_player_id}_total_player_speed"]+=speed_opponent
    current_player_stats[f"player_{opponent_player_id}_last_player_speed"]=speed_opponent
    # print(current_player_stats)
    player_stats_data.append(current_player_stats)

  player_stats_data_df=pd.DataFrame(player_stats_data)
  frames_df=pd.DataFrame({"frame_num":list(range(len(video_frames)))})

  player_stats_data_df = pd.merge(frames_df, player_stats_data_df, left_on="frame_num", right_on="frame_number", how="left")
  player_stats_data_df = player_stats_data_df.drop(columns=["frame_number"])
  player_stats_data_df = player_stats_data_df.ffill()


  player_stats_data_df["player_1_average_shot_speed"]=player_stats_data_df["player_1_total_shot_speed"]/player_stats_data_df["player_1_number_of_shots"]
  player_stats_data_df["player_2_average_shot_speed"]=player_stats_data_df["player_2_total_shot_speed"]/player_stats_data_df["player_2_number_of_shots"]
  player_stats_data_df["player_1_average_player_speed"]=player_stats_data_df["player_1_total_player_speed"]/player_stats_data_df["player_2_number_of_shots"]
  player_stats_data_df["player_2_average_player_speed"]=player_stats_data_df["player_2_total_player_speed"]/player_stats_data_df["player_1_number_of_shots"]


  #Draw bounding boxes
  output_video_frames=player_tracker.draw_bboxes(video_frames,player_detections)
  output_video_frames=ball_tracker.draw_bboxes(output_video_frames,ball_detections)
  output_video_frames=court_line_detector.draw_keypoints_on_video(output_video_frames,court_keypoints)


  #Draw mini court
  output_video_frames=mini_court.draw_mini_court(output_video_frames)

  output_video_frames=mini_court.draw_points_on_mini_court(output_video_frames,player_mini_court_detections)
  output_video_frames=mini_court.draw_points_on_mini_court(output_video_frames,ball_mini_court_detections,color=(0,255,0))

  output_video_frames=draw_player_stats(output_video_frames,player_stats_data_df)

  #Draw frame number
  for i,frame in enumerate(output_video_frames):
    cv2.putText(frame, f"Frame : {i}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

  #Save video
  save_video(output_video_frames,"output_videos/output_video.avi",fps=fps)

if __name__ == "__main__":
  main()