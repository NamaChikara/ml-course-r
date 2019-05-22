lrCostFunction <- function(X, y, lambda) {
  #lrCostFunction Compute cost for logistic regression with
  #regularization
  #   J <- lrCostFunction(X, y, lambda)(theta) computes the cost of using
  #   theta as the parameter for regularized logistic regression.
  
  function(theta) {
    # Initialize some useful values
    m <- length(y) # number of training examples
    
    # You need to return the following variables correctly
    J <- 0
    
    # ----------------------- YOUR CODE HERE -----------------------
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X %*% theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #
    
    h <- sigmoid(X %*% theta)
    
    # NOTE:
    # We cannot do
    #   costFunction <- (1 / m) * (-1 * t(y) %*% log(h) - t(1 - y) %*% log(1 - h))
    # since we lose decimals of percision.
    costFunction <- (-1 * t(y) %*% log(h) - t(1 - y) %*% log(1 - h)) / m
    
    # NOTE: the bias term, theta_0, is excluded.
    regTerm <- lambda * (t(theta[-1]) %*% theta[-1]) / (2 * m)
    
    costFunction + regTerm 
    # --------------------------------------------------------------
  }
}

lrGradFunction <- function(X, y, lambda) {
  #lrGradFunction Compute  gradient for logistic regression with
  #regularization
  #   J <- lrGradFunction( X, y, lambda)(theta) computes the
  #   gradient of the cost w.r.t. to the parameters.
  function(theta) {
    # Initialize some useful values
    m <- length(y) # number of training examples
    
    # You need to return the following variables correctly
    
    grad <- matrix(0,length(theta))
    
    # ----------------------- YOUR CODE HERE -----------------------
    # Instructions: set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    #
    # Hint: The computation of the cost function and gradients can be
    #       efficiently vectorized. For example, consider the computation
    #
    #           sigmoid(X %*% theta)
    #
    #       Each row of the resulting matrix will contain the value of the
    #       prediction for that example. You can make use of this to vectorize
    #       the cost function and gradient computations.
    #
    # Hint: When computing the gradient of the regularized cost function,
    #       there're many possible vectorized solutions, but one solution
    #       looks like:
    #           grad <- (unregularized gradient for logistic regression)
    #           temp <- theta
    #           temp(1) <- 0;   # because we don't add anything for j <- 0
    #           grad <- grad + YOUR_CODE_HERE (using the temp variable)
    #
    #
    h <- sigmoid(X %*% theta)
    
    grad <- (t(X) %*% (h - y)) / m
    
    # NOTE: the bias term, theta_0, is excluded.
    regTerm <- (lambda / m) * c(0, theta[-1])
    
    grad + regTerm
    
    # --------------------------------------------------------------
  }
}
