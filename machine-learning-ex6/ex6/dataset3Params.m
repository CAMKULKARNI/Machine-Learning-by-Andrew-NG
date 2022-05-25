function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_val = [0.01, 0.03, 0.10, 0.30, 1.00, 3.00, 10.00, 30.00, 100.00]
sigma_val = [0.01, 0.03, 0.10, 0.30, 1.00, 3.00, 10.00, 30.00, 100.00]
predicted_matrix = zeros(9, 9)

for i = 1:length(C_val)
    for j = 1:length(sigma_val)
        model = svmTrain(X, y, C_val(i), @(x1, x2) gaussianKernel(x1, x2, sigma_val(j)));
        prediction = svmPredict(model, Xval)
        predicted_matrix(i, j) = mean(double(prediction ~= yval))
    end
end

[colMin, row] = min(predicted_matrix);
[rowMin, col] = min(predicted_matrix');

[colMin, colIndex] = min(min(predicted_matrix)); 
[minValue, rowIndex] = min(predicted_matrix(:,colIndex))

C = C_val(rowIndex)
sigma = sigma_val(colIndex)




% =========================================================================

end
