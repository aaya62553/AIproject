from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
  def __init__(self,model_path):
    self.model=YOLO(model_path)

  def interpolate_ball_positrions(self,ball_positions):
    ball_positions=[x.get(1,[]) for x in ball_positions]
    df_ball_positions=pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])
    df_ball_positions=df_ball_positions.interpolate(method="linear")
    df_ball_positions=df_ball_positions.bfill()

    ball_positions=[{1:x} for x in df_ball_positions.to_numpy().tolist()]
    return ball_positions
  
  def get_ball_shot_frames(self,ball_positions):
    ball_positions=[x.get(1,[]) for x in ball_positions]
    df_ball_positions=pd.DataFrame(ball_positions,columns=["x1","y1","x2","y2"])
    df_ball_positions["mid_y"]=(df_ball_positions["y1"]+df_ball_positions["y2"])/2
    df_ball_positions["mid_y_rolling_mean"]=df_ball_positions["mid_y"].rolling(window=5,min_periods=1,center=False).mean()
    df_ball_positions["delta_y"]=df_ball_positions["mid_y_rolling_mean"].diff()
    df_ball_positions["ball_hit"]=0
    minimum_change_frames_for_hit=25
    #Parcourir les positions de la balle
    for i in range(1,len(df_ball_positions)-int(minimum_change_frames_for_hit*1.2)):
        # Si la balle change de direction
        negative_position_change=df_ball_positions["delta_y"].iloc[i]>0 and df_ball_positions["delta_y"].iloc[i+1]<0
        positive_position_change=df_ball_positions["delta_y"].iloc[i]<0 and df_ball_positions["delta_y"].iloc[i+1]>0
        if negative_position_change or positive_position_change:
            change_count=0 #Represente le nombre de frames (max 25) ou la balle continue dans sa direction
            # Parcourir les frames suivantes
            for change_frame in range(i+1,i+int(minimum_change_frames_for_hit*1.2)+1):

                # Tant que la balle continue dans la meme direction
                negative_position_change_following_frame=df_ball_positions["delta_y"].iloc[i]>0 and df_ball_positions["delta_y"].iloc[change_frame]<0
                positive_position_change_following_frame=df_ball_positions["delta_y"].iloc[i]<0 and df_ball_positions["delta_y"].iloc[change_frame]>0
                if negative_position_change and negative_position_change_following_frame:
                    change_count+=1
                elif positive_position_change and positive_position_change_following_frame:
                    change_count+=1
        
            if change_count>=minimum_change_frames_for_hit-1:
                df_ball_positions["ball_hit"].iloc[i]=1
    frame_nums_with_ball_hits=df_ball_positions[df_ball_positions["ball_hit"]==1].index.tolist()
    #Le but de l'algo est de vérifier que la balle continue dans la meme direction pendant au moins 25 frames
    #Dans ce cas, on considère que la balle a été frappée
    #Sinon on considère que la balle n'a pas été frappée et on continue
    return frame_nums_with_ball_hits
  
  def detect_frames(self,frames,read_from_stub=False,stub_path=None):
    ball_detections=[]
    if read_from_stub and stub_path is not None:
      with open(stub_path,"rb") as f:
        ball_detections=pickle.load(f)
      return ball_detections

    for frame in frames:
      ball_dict=self.detect_frame(frame)
      ball_detections.append(ball_dict)

    if stub_path is not None:
      with open(stub_path,"wb") as f:
        pickle.dump(ball_detections,f)
    return ball_detections

  def detect_frame(self,frame):

    results=self.model.predict(frame,conf=0.15)[0] # persist=True means that the tracker will remember the object's position in the next frame
    ball_dict={}
    for box in results.boxes:
      result=box.xyxy.tolist()[0]
      ball_dict[1]=result
    return ball_dict
  
  def draw_bboxes(self,video_frames,ball_detections):
    output_video_frames=[]
    for frame,ball_dict in zip(video_frames,ball_detections): #loop 2 lists at the same time
      for track_id,result in ball_dict.items():
        x1,y1,x2,y2=result 
        cv2.putText(frame, f"Ball ID : {track_id}", (int(result[0]), int(result[1] -10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2) #put text on the frame
        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)  
      output_video_frames.append(frame)
    return output_video_frames
