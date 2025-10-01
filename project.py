import mediapipe as mp
import cv2
import numpy as np
from PIL import Image,ImageSequence

def hand_rec():    
     current_gesture = None
     gif_frames = []
     gif_index = 0
     frame_timer = 0 

     def thumbs_up(landmark): #landmark specification for thumbs up
          thumb_tip = landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
          thumb_ip = landmark.landmark[mp_hands.HandLandmark.THUMB_IP]
          index_tip = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
          middle_tip = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

          thumb_extended = thumb_ip.y > thumb_tip.y
          finger_fold = index_tip.y > thumb_ip.y and middle_tip.y > thumb_ip.y

          if thumb_extended and finger_fold:
               return True
          return False

     def peace_sign(landmark): #landmark specification for peace sign
          thumb_tip = landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
          ring_ip = landmark.landmark[mp_hands.HandLandmark.RING_FINGER_DIP]
          index_tip = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
          index_ip = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
          middle_tip = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
          middle_ip = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]

          fingers_extended = index_ip.y > index_tip.y and middle_ip.y > middle_tip.y
          finger_fold = middle_ip.y < ring_ip.y and thumb_tip.y < ring_ip.y

          if finger_fold and fingers_extended:
               return True
          return False

     def middle_finger(landmark): #landmark specification middle finger
          index_tip = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
          index_pip = landmark.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]

          ring_tip = landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
          ring_pip = landmark.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]

          pinky_tip = landmark.landmark[mp_hands.HandLandmark.PINKY_TIP]
          pinky_pip = landmark.landmark[mp_hands.HandLandmark.PINKY_PIP]

          middle_ip_2 = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
          middle_tip = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]


          finger_fold = index_tip.y > index_pip.y and ring_tip.y > ring_pip.y and pinky_tip.y > pinky_pip.y
          middle_extend = middle_tip.y < middle_ip_2.y

          if finger_fold and middle_extend:
               return True
          return False
          
     def thumbs_down(landmark): #landmark specification for thumbs down
          thumb_tip = landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
          thumb_ip = landmark.landmark[mp_hands.HandLandmark.THUMB_IP]
          ring_tip = landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
          middle_tip = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

          thumb_extended = thumb_ip.y < thumb_tip.y
          finger_fold = ring_tip.y < middle_tip.y 

          if thumb_extended and finger_fold:
               return True
          return False

     def rock_sign(landmark): #landmark specification for rock sign
          thumb_tip = landmark.landmark[mp_hands.HandLandmark.THUMB_TIP]
          ring_tip = landmark.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
          middle_tip = landmark.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

          finger_fold = middle_tip.y > thumb_tip.y and ring_tip.y > thumb_tip.y

          if finger_fold:
               return True
          return False


     mp_drawing = mp.solutions.drawing_utils    #setting up mediapipe solutions
     mp_hands = mp.solutions.hands

     with (mp_hands.Hands(min_detection_confidence = 0.9, min_tracking_confidence=0.7) as hands
          ):    
          cap =cv2.VideoCapture(0)
          while cap.isOpened():
               ret, frame = cap.read()            #setting up opencv
               frame=cv2.resize(frame,(800,600))

               image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
               image = cv2.flip(image,1)
               image.flags.writeable = False
               hand_results = hands.process(image)  #processing video results
               image.flags.writeable = True
               image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

               if hand_results.multi_hand_landmarks:    #setting up hand landmarks
                    for num, hand in enumerate(hand_results.multi_hand_landmarks):
                         mp_drawing.draw_landmarks(image,hand,mp_hands.HAND_CONNECTIONS,
                                                  mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=4),
                                                  mp_drawing.DrawingSpec(color=(121,44,250),thickness=2,circle_radius=2)

                                                  )
               
               if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                         if thumbs_up(hand_landmarks):
                            if current_gesture != "thumbs_up":
                              current_gesture = "thumbs_up"
                              gif = Image.open("thumbs_up_gif.gif")
                              gif_frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]    #triggering thumbs up sign
                              gif_index = 0

                            frame_timer += 1
                            if frame_timer >= 1:
                                   gif_index = (gif_index + 1) % len(gif_frames)
                                   frame_timer = 0

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            gif_frame = gif_frames[gif_index].resize((200, 300))
                            image_pil.paste(gif_frame, (500, 100), gif_frame)
                            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                    
                         elif peace_sign(hand_landmarks):
                            if current_gesture != "peace_sign":
                              current_gesture = "peace_sign"
                              gif = Image.open("peace_sign_gif.gif")
                              gif_frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]   #triggering peace sign
                              gif_index = 0

                            frame_timer += 1
                            if frame_timer >= 1:
                                   gif_index = (gif_index + 1) % len(gif_frames)
                                   frame_timer = 0

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            gif_frame = gif_frames[gif_index].resize((200, 300))
                            image_pil.paste(gif_frame, (500, 100), gif_frame)
                            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                         
                         elif thumbs_down(hand_landmarks):
                            if current_gesture != "thumbs_down":
                              current_gesture = "thumbs_down"
                              gif = Image.open("thumbs_down_gif.gif")
                              gif_frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]   #triggering thumbs down sign
                              gif_index = 0

                            frame_timer += 1
                            if frame_timer >= 1:
                                   gif_index = (gif_index + 1) % len(gif_frames)
                                   frame_timer = 0

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            gif_frame = gif_frames[gif_index].resize((200, 300))
                            image_pil.paste(gif_frame, (500, 100), gif_frame)
                            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                              
                         elif middle_finger(hand_landmarks):
                            if current_gesture != "middle_finger":
                              current_gesture = "middle_finger"
                              gif = Image.open("middle_finger_gif.gif")
                              gif_frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]   #triggering middle finger sign
                              gif_index = 0

                            frame_timer += 1
                            if frame_timer >= 1:
                                   gif_index = (gif_index + 1) % len(gif_frames)
                                   frame_timer = 0

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            gif_frame = gif_frames[gif_index].resize((200, 300))
                            image_pil.paste(gif_frame, (500, 100), gif_frame)
                            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

                         elif rock_sign(hand_landmarks):
                            if current_gesture != "rock_sign":
                              current_gesture = "rock_sign"
                              gif = Image.open("rock_sign_gif.gif")
                              gif_frames = [frame.copy().convert("RGBA") for frame in ImageSequence.Iterator(gif)]   #triggering rock sign
                              gif_index = 0

                            frame_timer += 1
                            if frame_timer >= 1:
                                   gif_index = (gif_index + 1) % len(gif_frames)
                                   frame_timer = 0

                            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            image_pil = Image.fromarray(image_rgb)
                            gif_frame = gif_frames[gif_index].resize((200, 300))
                            image_pil.paste(gif_frame, (500, 100), gif_frame)
                            image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
          
               cv2.putText(image,"CamRead",(290,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),4)                               
               cv2.imshow('CamRead - Filtered Fun',image)

               if cv2.waitKey(10) & 0xFF == ord('q'): #breaking opencv cam
                         break


     cap.release()
     cv2.destroyAllWindows()   
     print(mp_hands.HAND_CONNECTIONS)       

def interface():
     import tkinter as tk   #setting up basic tkinter welcome GUI   

     window = tk.Tk()
     window.title("CamRead")
     window.geometry("1000x600")
     logo = tk.PhotoImage(file="thumbs_up.png")
     window.iconphoto(window,logo)
     window.resizable(height=False,width=False)

     bg = tk.PhotoImage(file="bg_img.png")
     canvas = tk.Canvas(window,width=1000,height=600)
     canvas.pack()
     canvas.create_image(0,0,image=bg,anchor="nw")
     canvas.create_text(500,50,text="WELCOME TO CAMREAD!!!",font=("Arial",28),fill="#aebb3e")
     canvas.create_text(500,200,text="\t\t\t\t  HELLO THERE!\n " \
                                   "\t\t        WELCOME TO CAMREAD, A FUN SPACE TO \n\tUSE YOUR HANDS, MAKE FUN GESTURES AND TRIGGER MANY FUN ANIMATION!\n " \
                                   "\t\t        TO GET STARTED CLICK THE BUTTON BELOW!",font=("arial",12),fill="#2E343A")

     button = tk.Button(canvas,text="OPEN-CAM",command=hand_rec,font=("Arial",12),bg="#e4cba6",fg="Black")
     button.place(x=445,y=400)
     window.mainloop()


interface()

