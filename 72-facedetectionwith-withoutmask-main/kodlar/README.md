person even if he/she is wearing a mask or not. Steps:
1. Add images with mask and without mask in the dataset folder under respective name.
2. Create embedding for the face using command
'python extract_embeddings.py --dataset dataset  --embeddings output/embeddings.pickle  --detector face_detection_model  --embedding-model openface_nn4.small2.v1.t7'
3. Train the model so it recognizes each face at real time. Command:
'python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle'
4. Test the model if it recognizes faces correctly. Command:
'python recognize_video.py --detector face_detection_model --embedding-model openface_nn4.small2.v1.t7 --recognizer output/recognizer.pickle --le output/le.pickle'
