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

a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
b = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

min_error = 1;
cnt = 1;
for i =  1:size(a,2)
	for j = 1:size(b,2)
		fprintf('loop %d\n',cnt);
		c = a(i);
		s = b(j);
		model= svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s)); 
		predictions =svmPredict(model, Xval);
		err = mean(double(predictions ~= yval));
		fprintf('error %0.5f c = %0.5f s = %0.5f \n',err,c,s);
		if err < min_error
			C = c;
			sigma = s;
			min_error = err
			fprintf('min error so far %0.5f,   c = %0.5f s = %0.5f \n',err,c,s);

		end

		cnt = cnt+1;
	end
end

% =========================================================================

end
