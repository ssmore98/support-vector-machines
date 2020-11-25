function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%



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
% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
predictions = svmPredict(model, Xval);
mincost = mean(double(predictions ~= yval));
Cmin = C;
sigma_min = sigma;
C = 0.01;
for i = 1:3
    sigma = 0.01;
    for j = 1:3
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        cost = mean(double(predictions ~= yval));
        if cost < mincost
            Cmin = C;
            sigma_min = sigma;
            mincost = cost;
        end
        sigma = sigma * 10;
    end
    C = C * 10;
end

C = Cmin;
sigma = sigma_min;

% =========================================================================

end
