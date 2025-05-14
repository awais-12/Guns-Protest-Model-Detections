import cv2
from ultralytics import YOLO

def detect_and_display(image_path):
    # Load the model
    model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
    # Run inference
    results = model(image_path)
    
    # Get annotated image
    annotated_img = results[0].plot()
    
    # Display results
    cv2.imshow("Detection Results", annotated_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save results
    output_path = "detected_output.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"Results saved to {output_path}")
    return output_path

# Usage
detect_and_display("5.jpg")



# import cv2
# from ultralytics import YOLO

# def detect_media(input_path):
#     model = YOLO('./runs/detect/Normal_Compressed/weights/best.pt')
    
#     if input_path.lower().endswith(('.jpg', '.png', '.jpeg')):
#         # Image processing
#         results = model(input_path)
#         annotated = results[0].plot()
#         cv2.imshow("Results", annotated)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         cv2.imwrite("detected_image.jpg", annotated)
        
#     elif input_path.lower().endswith(('.mp4', '.avi', '.mov')):
#         # Video processing
#         cap = cv2.VideoCapture(input_path)
#         writer = None
        
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
                
#             results = model(frame)
#             annotated = results[0].plot()
            
#             if writer is None:  # Initialize writer on first frame
#                 h, w = annotated.shape[:2]
#                 writer = cv2.VideoWriter('output_video.avi', 
#                                        cv2.VideoWriter_fourcc(*'XVID'),
#                                        30, (w, h))
#             writer.write(annotated)
            
#             cv2.imshow("Live Detection", annotated)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#         cap.release()
#         if writer:
#             writer.release()
#         cv2.destroyAllWindows()
        
#     else:
#         print("Unsupported file format")

# # Usage for video
# detect_media("your_video.mp4")

