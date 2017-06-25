elapsedTime = [];
car_vector = [];
face_vector = [];
correct_vectors = [];
for k = 1:1:10
    k
    tic
    [FOREST, C, BOW_matrix_cars, BOW_matrix_faces] = mytrainingSPM(k);
    [correct_car, correct_face, correctness] = mytestingSPM(FOREST, C, BOW_matrix_cars, BOW_matrix_faces, k); 
    elapsedTime = [elapsedTime toc]
    car_vector = [car_vector correct_car];
    face_vector = [face_vector correct_face];
    correct_vectors = [correct_vectors correctness];
end
subplot(1,2,1)
plot(1:1:10,elapsedTime,'b-*');
hold on
title('Elapsed time');
xlabel('Vocab length');
ylabel('Time(s)');
subplot(1,2,2)
plot(1:1:10,car_vector./50,'r-*');
hold on
plot(1:1:10,face_vector./50,'b-*');
plot(1:1:10,correct_vectors,'g-*');
title('Bag of Words');
xlabel('Vocab length');
ylabel('Accuracy');