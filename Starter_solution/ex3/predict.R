predict <- function(Theta1, Theta2, X) {
  #PREDICT Predict the label of an input given a trained neural network
  #   p <- PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
  #   trained weights of a neural network (Theta1, Theta2)
  
  # Useful values
  if (is.vector(X))
    X <- t(X)
  
  m <- dim(X)[1]
  num_labels <- dim(Theta2)[1]
  
  # You need to return the following variables correctly
  p <- rep(0,m)
  
  # ----------------------- YOUR CODE HERE -----------------------
  # Instructions: Complete the following code to make predictions using
  #               your learned neural network. You should set p to a
  #               vector containing labels between 1 to num_labels.
  #
  # Hint: The max function might come in useful. In particular, the which.max
  #       function can return the index of the max element, for more
  #       information see '?which.max'. If your examples are in rows, then, you
  #       can use apply(A, 1, max) to obtain the max for each row.
  #
  
  # Suppose dim(X) = 5000 x 400, dim(Theta1) = 25 x 401, dim(Theta2) = 10 x 26.
  # I.e. X has its features in rows, and the Theta1 weights for those features
  # are also in rows. Thus, calculating Theta1 %*% X will not work, we need to
  # do Theta1 %*% t(X).  The output of Theta1 %*% t(X) is then 25 x 5000 -- 
  # the same step needs to be taken when calculating Theta2.
  # 
  # Process:
  #   1) Set a(1) = [1, X] (add the bias layer)
  #   2) Calculate z(2) = Theta1 %*% t(a(1)) (calculate first hidden layer)
  #   3) Calculate a(2) = [1, g(z(2))] (calculate input to second hidden layer)
  #   4) Calculate z(3) = Theta2 %*% t(a(2)) (calculate second hidden layer)
  #   5) Calculate a(3) = g(z(3)) = h_theta(x) (calculate output layer)
  
  a1 <- cbind(1, X)
  z2 <- Theta1 %*% t(a1)
  z2 <- t(z2)
  a2 <- cbind(1, sigmoid(z2))
  z3 <- Theta2 %*% t(a2)
  z3 <- t(z3)
  a3 <- sigmoid(z3)
  apply(a3, 1, which.max)

  # --------------------------------------------------------------------------
}
